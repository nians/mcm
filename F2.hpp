/*	MCM file compressor

  MCM-F2: asymmetric fast-mode codec.

  Decode side is copies plus table-driven adaptive rANS only -- no per-bit
  arithmetic coding, no context mixing -- so it runs at memory speed instead of
  model speed. Encode side owns all the modeling decisions (parse, contexts),
  mirroring the plan in doc/fastmode_optimization_plan_zh.md (phase P2).

  Format (per compress() call, i.e. per solid block):
    block   := chunk*
    chunk   := leb128 payload_len, payload
    payload := leb128 raw_len, u8 mode, body
    mode 1  := raw_len stored bytes
    mode 0  := leb128 n_segments, segment*
    segment := leb128 n_tokens, leb128 data_len, rans_data
  Tokens (decoded in order, all symbols nibble-coded through adaptive CDFs):
    LIT               : one literal byte (hi/lo nibble, o2-lite + post-match ctx)
    MATCH             : length (>=4), distance (slot + raw bits); reps shift
    REP0/REP1/REP2    : length (>=2), distance from the rep list (move-to-front)
  rANS: 32-bit state, 8-bit renorm, 12-bit probabilities, encoded backwards
  per segment so the models adapt strictly forward on both sides.

  LICENSE: GPLv3, same as the rest of MCM (see COPYING).
*/

#ifndef _F2_HPP_
#define _F2_HPP_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

// SSE2 is baseline on x86-64; every 16-lane model loop below (CDF blend,
// symbol lookup, model update) has a hand-written vector path because the
// autovectorizer takes none of them (short trip counts, data-dependent exits).
// The vector paths are bit-exact with the scalar ones -- encoder and decoder
// models must stay in lockstep no matter which build runs on which side.
#if !defined(F2_FORCE_SCALAR) && (defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define F2_SSE2 1
#include <emmintrin.h>
#endif

// F2_NO_ARCHIVE builds only the core codec (standalone tests) without the
// embedded-CM selector wrapper and its heavy dependency chain.
#ifndef F2_NO_ARCHIVE
#include "CM-inl.hpp"
#endif
#include "Compressor.hpp"
#include "Stream.hpp"
#include "Util.hpp"

namespace f2 {

static const size_t kChunkSize = 16 * MB;  // window == chunk == selector granularity
static const size_t kSegTokens = 128 * 1024;
#ifdef F2_PROB_BITS
static const uint32_t kProbBits = F2_PROB_BITS;
#else
static const uint32_t kProbBits = 12;
#endif
static const uint32_t kProbScale = 1u << kProbBits;
static const uint32_t kRansL = 1u << 23;  // lower bound of the state interval
static const size_t kMinMatch = 4;
static const size_t kMinRep = 2;
static const size_t kMaxLenDirect = 13;   // length slots 0..13 are direct
static const size_t kLenBypass6 = 14;     // slot 14: 6 raw bits
static const size_t kLenBypass16 = 15;    // slot 15: 16 raw bits
static const size_t kMaxMatchLen = 65535;
static const size_t kNumReps = 3;

// Token kinds (symbols of the token model).
enum TokKind {
  kTokLit = 0, kTokMatch = 1, kTokRep0 = 2, kTokRep1 = 3, kTokRep2 = 4,
  kTokLzp = 5,
  kTokKinds = 6,
};

// LZP layer: both sides maintain "last position whose preceding 4 bytes hashed
// here" and verify the actual context bytes, so a hash collision simply means
// no candidate on either side. A hit is content-addressed -- only the length
// is coded, no distance -- which is where the CM fast mode gets most of its
// win on redundant data (study section P2b).
static const uint32_t kLzpBits = 18;
static const size_t kMinLzpLen = 2;
static const uint32_t kLzpNil = 0xFFFFFFFFu;
static const uint16_t kNoPred = 0xFFFF;

// Branch-free symbol lookup over a 17-entry nibble CDF: the predicate
// cdf[i] <= slot holds on a prefix (strict monotonicity), so the popcount of
// the compare mask IS the symbol. cdf[16] == kProbScale > slot caps it at 15.
// Replaces a data-dependent while loop whose mispredicts dominate on
// high-entropy nibbles.
ALWAYS_INLINE size_t LookupCdf(const uint16_t* cdf, uint32_t slot) {
#ifdef F2_SSE2
  const __m128i sv = _mm_set1_epi16(static_cast<short>(slot + 1));
  const __m128i m0 = _mm_cmpgt_epi16(
      sv, _mm_loadu_si128(reinterpret_cast<const __m128i*>(cdf + 1)));
  const __m128i m1 = _mm_cmpgt_epi16(
      sv, _mm_loadu_si128(reinterpret_cast<const __m128i*>(cdf + 9)));
  const unsigned mask =
      static_cast<unsigned>(_mm_movemask_epi8(_mm_packs_epi16(m0, m1)));
#ifdef _MSC_VER
  return static_cast<size_t>(__popcnt(mask));
#else
  return static_cast<size_t>(__builtin_popcount(mask));
#endif
#else
  size_t sym = 0;
  while (cdf[sym + 1] <= slot) ++sym;
  return sym;
#endif
}

#ifdef F2_TRACE
#include <cstdio>
// Debug-only dual traces: encoder and decoder each append one line per token
// with the full context tuple; diffing the two files pinpoints the first
// state divergence. Never enabled in production builds.
inline FILE* F2TraceFile(int which) {
  static FILE* f[2] = {nullptr, nullptr};
  if (!f[which]) f[which] = fopen(which ? "f2_dec.trace" : "f2_enc.trace", "w");
  return f[which];
}
#endif

class LzpTable {
public:
  // Order-4 context. Order-6 was measured slightly worse across every corpus:
  // in an LZ frame the rep list and explicit matches already cover most of
  // what higher-order LZP would add.
  void Reset() { table_.assign(size_t(1) << kLzpBits, kLzpNil); }

  // Candidate for position pos given the buffer decoded so far, or kLzpNil.
  ALWAYS_INLINE uint32_t Candidate(const uint8_t* buf, size_t pos) const {
    if (pos < 4) return kLzpNil;
    const uint32_t c = table_[Hash(buf, pos)];
    if (c == kLzpNil) return kLzpNil;
    // Verify the context so both sides agree on collision misses.
    uint32_t a, b;
    memcpy(&a, buf + pos - 4, 4);
    memcpy(&b, buf + c - 4, 4);
    return a == b ? c : kLzpNil;
  }

  ALWAYS_INLINE void Insert(const uint8_t* buf, size_t pos) {
    if (pos >= 4) table_[Hash(buf, pos)] = static_cast<uint32_t>(pos);
  }

private:
  ALWAYS_INLINE static uint32_t Hash(const uint8_t* buf, size_t pos) {
    uint32_t v;
    memcpy(&v, buf + pos - 4, 4);
    return (v * 2654435761u) >> (32 - kLzpBits);
  }

  std::vector<uint32_t> table_;
};

// ---------------------------------------------------------------------------
// Adaptive nibble model: a 16-symbol CDF over 12-bit probability space that
// adapts toward the step function of each observed symbol. Every frequency
// stays >= 1 by construction (targets keep unit gaps), so any nibble stays
// encodable at all times.
class NibModel {
public:
  uint16_t cdf_[17];
  uint16_t count_ = 0;

  NibModel() { Init(); }

  void Init() {
    for (int i = 0; i <= 16; ++i) cdf_[i] = static_cast<uint16_t>(i * (kProbScale / 16));
    count_ = 0;
  }

  ALWAYS_INLINE uint32_t Start(size_t sym) const { return cdf_[sym]; }
  ALWAYS_INLINE uint32_t Freq(size_t sym) const { return cdf_[sym + 1] - cdf_[sym]; }

  ALWAYS_INLINE size_t Lookup(uint32_t slot) const {
    return LookupCdf(cdf_, slot);
  }

  ALWAYS_INLINE void Update(size_t sym) {
    const int rate = count_ < 16 ? 3 : (count_ < 64 ? 4 : 5);
    count_ += count_ < 64;
#ifdef F2_SSE2
    // Lanes 1..16 uniformly: for i > sym the target i + (kProbScale-16)
    // equals the scalar kProbScale - (16-i); lane 16 computes target
    // kProbScale and a zero step, so cdf_[16] stays pinned and there is no
    // tail. _mm_sra_epi16 is the same arithmetic shift the scalar >> does.
    const __m128i bump = _mm_set1_epi16(static_cast<short>(kProbScale - 16));
    const __m128i sv = _mm_set1_epi16(static_cast<short>(sym));
    const __m128i cnt = _mm_cvtsi32_si128(rate);
    const __m128i i0 = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
    const __m128i i1 = _mm_setr_epi16(9, 10, 11, 12, 13, 14, 15, 16);
    __m128i c0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(cdf_ + 1));
    __m128i c1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(cdf_ + 9));
    const __m128i t0 =
        _mm_add_epi16(i0, _mm_and_si128(_mm_cmpgt_epi16(i0, sv), bump));
    const __m128i t1 =
        _mm_add_epi16(i1, _mm_and_si128(_mm_cmpgt_epi16(i1, sv), bump));
    c0 = _mm_add_epi16(c0, _mm_sra_epi16(_mm_sub_epi16(t0, c0), cnt));
    c1 = _mm_add_epi16(c1, _mm_sra_epi16(_mm_sub_epi16(t1, c1), cnt));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cdf_ + 1), c0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cdf_ + 9), c1);
#else
    for (int i = 1; i < 16; ++i) {
      const int target = i <= static_cast<int>(sym)
        ? i
        : static_cast<int>(kProbScale) - (16 - i);
      cdf_[i] = static_cast<uint16_t>(cdf_[i] + ((target - static_cast<int>(cdf_[i])) >> rate));
    }
#endif
  }
};

// ---------------------------------------------------------------------------
// Two-model CDF blend: out[i] = a[i] + ((b[i]-a[i])*w >> 8), w in [0,256).
// For monotone inputs with unit gaps the floor error cancels against the
// preserved gap (gap_out >= min(gap_a, gap_b) >= 1), so the blended CDF stays
// strictly monotone and every nibble stays encodable. This is the mixing that
// a lone sparse high-order context lacks: cold buckets keep w low and merely
// shadow the dense model instead of hurting it.
static const uint32_t kBlendBits = 16;   // high-order bucket count = 64K
static const int kBlendInitW = 64;       // start at 25% high-order weight
static const int kBlendMinW = 8;
static const int kBlendMaxW = 248;
// The cascaded o4 stage starts near-mute and only speaks once its bucket has
// seen data: o4 buckets are an order sparser than o3 ones, and at the o3
// stage's 25% initial weight their cold uniform CDFs tax every literal (~+1%
// measured). Both sides branch on the model's own count, so the gate is
// lockstep by construction.
static const int kBlend2InitW = 16;
// Evidence threshold before the o4 stage may speak: below this many
// observations the bucket's CDF is still in its crudest adaptation regime and
// blending it in is pure noise tax.
#ifdef F2_O4_GATE
static const uint16_t kBlend2Gate = F2_O4_GATE;
#else
static const uint16_t kBlend2Gate = 16;
#endif

// Lagrangian decode-speed bias, in 1/16-bit units per decode event: the DP
// minimizes bits + bias * events, so fragmenting into many small tokens must
// pay for its decode cost. 0 = pure rate optimization.
#ifdef F2_SPEED_BIAS
static const uint32_t kSpeedBias = F2_SPEED_BIAS;
#else
static const uint32_t kSpeedBias = 4;
#endif
// Literal coder tier, carried per chunk in a header byte so the decoder (which
// has no compile-time lambda of its own) simply obeys the encoder's choice:
//   0 = two-model CDF blend            (speed corner)
//   1 = + o4 cascade with shadow ledger (intermediate; F2_LIT_TIER=1)
//   2 = binary-mixed literals           (rate corner: the mini-CM tier)
// Rate-leaning builds take tier 2; speed-leaning builds ship tier 0.
#ifdef F2_LIT_TIER
static const int kLitTier = F2_LIT_TIER;
#else
static const int kLitTier = kSpeedBias < 3 ? 2 : 0;
#endif
static const bool kUseO4 = kLitTier == 1;

ALWAYS_INLINE void BlendCdf(const uint16_t* a, const uint16_t* b, uint32_t w,
                            uint16_t* out) {
#ifdef F2_SSE2
  // a + ((b-a)*w >> 8) == (a*(256-w) + b*w) >> 8 exactly: a*256 is a multiple
  // of 256, so the floor of the sum splits at a. madd computes the pair
  // product in i32 lanes with no overflow (products <= kProbScale*256 = 2^20).
  // Lane 16 blends to kProbScale on its own since a[16] == b[16] == kProbScale.
  const __m128i wv =
      _mm_set1_epi32(static_cast<int>((w << 16) | (256u - w)));
  const __m128i va0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 1));
  const __m128i vb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 1));
  const __m128i va1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 9));
  const __m128i vb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 9));
  const __m128i p00 = _mm_madd_epi16(_mm_unpacklo_epi16(va0, vb0), wv);
  const __m128i p01 = _mm_madd_epi16(_mm_unpackhi_epi16(va0, vb0), wv);
  const __m128i p10 = _mm_madd_epi16(_mm_unpacklo_epi16(va1, vb1), wv);
  const __m128i p11 = _mm_madd_epi16(_mm_unpackhi_epi16(va1, vb1), wv);
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 1),
      _mm_packs_epi32(_mm_srli_epi32(p00, 8), _mm_srli_epi32(p01, 8)));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 9),
      _mm_packs_epi32(_mm_srli_epi32(p10, 8), _mm_srli_epi32(p11, 8)));
  out[0] = 0;
#else
  out[0] = 0;
  out[16] = static_cast<uint16_t>(kProbScale);
  for (int i = 1; i < 16; ++i) {
    const int32_t ai = a[i];
    out[i] = static_cast<uint16_t>(ai + (((b[i] - ai) * static_cast<int32_t>(w)) >> 8));
  }
#endif
}

// Weight adapts toward "how often the high-order model assigns the observed
// symbol more probability" (binary target measured better than a three-level
// proportional one on every corpus).
ALWAYS_INLINE void UpdateBlendWeight(uint8_t* w, uint32_t fa, uint32_t fb) {
  const int target = fb >= fa ? 256 : 0;
  int nw = *w + ((target - *w) >> 5);
  if (nw < kBlendMinW) nw = kBlendMinW;
  if (nw > kBlendMaxW) nw = kBlendMaxW;
  *w = static_cast<uint8_t>(nw);
}

// Stage-2 variant, two differences from the o3 rule. Floor 0: a muted o4
// stage costs exactly nothing (w=0 reproduces the two-model mix bit for bit)
// and the next agreeing symbol lifts it back to 8. Evidence test: fb >= fa
// alone misfires on flat buckets -- whenever the base under-predicts a symbol
// a near-uniform c "wins" by default, which is exactly what converged o4
// buckets look like on high-entropy literals (machine code), so the weight
// climbs on the worst possible data. Demanding fb above uniform means the o4
// model only gains weight where it is genuinely concentrated on the outcome.
ALWAYS_INLINE void UpdateBlendWeight2(uint8_t* w, uint32_t fa, uint32_t fb) {
  const int target = (fb >= fa && fb > kProbScale / 16) ? 256 : 0;
  int nw = *w + ((target - *w) >> 5);
  if (nw > kBlendMaxW) nw = kBlendMaxW;
  *w = static_cast<uint8_t>(nw < 0 ? 0 : nw);
}



// Defined below Models; the segment coders' o4 ledger needs it early.
inline const uint16_t* CostTable();

// ---------------------------------------------------------------------------
// Binary-mixed literal tier (rate corner). Literals are coded MSB-first as 8
// binary decisions; each bit's probability is a learned logistic-domain mix
// of five predictors (bias/o1/o2-hash/o4-hash/predicted-byte). This is a
// miniature of the CM mixer -- the piece the study identified as the actual
// source of the remaining rate gap -- confined to literals and to chunks
// whose header opts in, so the speed corner never pays for it.
//
// All arithmetic is integer and identical on both sides. The stretch/squash
// tables are built once from libm doubles; encoder and decoder of one archive
// run in one process, and cross-build decode additionally relies on libm
// giving the same rounding for these 4096 points (true in practice for glibc;
// the F2 format is still branch-internal).
static const int kStClamp = 2047;         // stretch output clamp
static const int kPClamp = 4096;          // probability scale (reuses kProbScale)

inline const int16_t* StretchTab() {
  static std::vector<int16_t> t;
  if (t.empty()) {
    t.resize(kPClamp);
    for (int p = 0; p < kPClamp; ++p) {
      const double pp = (p < 1 ? 1 : p) / 4096.0;
      const int v = static_cast<int>(std::lround(std::log(pp / (1.0 - pp)) * 256.0));
      t[p] = static_cast<int16_t>(v < -kStClamp ? -kStClamp : (v > kStClamp ? kStClamp : v));
    }
  }
  return t.data();
}

inline const uint16_t* SquashTab() {
  static std::vector<uint16_t> t;
  if (t.empty()) {
    t.resize(2 * kStClamp + 2);
    for (int i = 0; i < 2 * kStClamp + 2; ++i) {
      const double x = (i - kStClamp - 1) / 256.0;
      int v = static_cast<int>(std::lround(4096.0 / (1.0 + std::exp(-x))));
      if (v < 1) v = 1;
      if (v > 4095) v = 4095;
      t[i] = static_cast<uint16_t>(v);
    }
  }
  return t.data();
}

// Bit-history state machine counter: one byte per context slot encoding
// bounded (n0, n1) counts with opponent halving on each observation -- the
// nonstationarity trick shift counters cannot express (fresh contrary
// evidence discounts the accumulated majority instead of nudging it). The
// 16x16 grid gives exactly 256 states; probabilities live in a per-table
// StateMap (256 entries, L1-resident, shared across all context slots), so
// the big hashed tables shrink from 4 to 1 byte per slot -- a quarter of the
// DRAM traffic of the previous BitModel layout.
inline const uint8_t* StateNext() {
  static std::vector<uint8_t> t;
  if (t.empty()) {
    t.resize(512);
    for (int n0 = 0; n0 < 16; ++n0) {
      for (int n1 = 0; n1 < 16; ++n1) {
        const int s = n0 * 16 + n1;
        // bit 0 observed: n0 grows, n1 discounts once past 2.
        int a0 = n0 < 15 ? n0 + 1 : 15;
        int b0 = n1 <= 2 ? n1 : n1 / 2 + 1;
        t[2 * s + 0] = static_cast<uint8_t>(a0 * 16 + b0);
        // bit 1 observed: mirror.
        int a1 = n0 <= 2 ? n0 : n0 / 2 + 1;
        int b1 = n1 < 15 ? n1 + 1 : 15;
        t[2 * s + 1] = static_cast<uint8_t>(a1 * 16 + b1);
      }
    }
  }
  return t.data();
}

// Dense low-order tables (o0/o1/pred, at most a few hundred KB) keep a
// direct per-slot adaptive probability: their slots are individually hot and
// the per-slot resolution is worth more than state compression. Only the
// sparse hashed tables ride the state machine + shared StateMap.
struct DirectBit {
  uint16_t p = 2048;
  uint16_t cnt = 0;
  ALWAYS_INLINE void Update(int bit) {
    const int rate = cnt < 32 ? 3 : (cnt < 256 ? 4 : 5);
    cnt += cnt < 256;
    const int target = bit ? 4095 : 1;
    p = static_cast<uint16_t>(p + ((target - static_cast<int>(p)) >> rate));
  }
};

// Adaptive probability per state, shared across every slot of one model
// table. Seeded from the state's implied Krichevsky-Trofimov estimate and
// refined online with a visit-staged rate.
struct StateMap {
  uint16_t p[256];
  uint16_t cnt[256];
  void Init() {
    for (int n0 = 0; n0 < 16; ++n0) {
      for (int n1 = 0; n1 < 16; ++n1) {
        const int v = 4096 * (2 * n1 + 1) / (2 * (n0 + n1) + 2);
        p[n0 * 16 + n1] = static_cast<uint16_t>(v < 1 ? 1 : (v > 4095 ? 4095 : v));
        cnt[n0 * 16 + n1] = 0;
      }
    }
  }
  ALWAYS_INLINE uint32_t P(uint8_t s) const { return p[s]; }
  ALWAYS_INLINE void Update(uint8_t s, int bit) {
    const int rate = cnt[s] < 64 ? 4 : (cnt[s] < 1024 ? 5 : 6);
    cnt[s] += cnt[s] < 1024;
    const int target = bit ? 4095 : 1;
    p[s] = static_cast<uint16_t>(p[s] + ((target - static_cast<int>(p[s])) >> rate));
  }
};

static const size_t kBinHashBits = 21;   // o2/o4 bit-model table size
#ifdef F2_MIX_LR
static const size_t kBinMixLR = F2_MIX_LR;
#else
static const size_t kBinMixLR = 11;              // mixer learning-rate shift
#endif
static const int32_t kBinWCap = 1 << 20;

struct BinLitModels {
  DirectBit o1[256 * 256];                 // [prev1][node]
  uint8_t o2[1u << kBinHashBits];
  uint8_t o3[1u << kBinHashBits];
  uint8_t o4[1u << kBinHashBits];
  uint8_t o6[1u << kBinHashBits];
  uint8_t word[1u << kBinHashBits];
  DirectBit pred[257 * 256];               // [predicted byte | 256=none][node]
  DirectBit o0[256];                       // [node]
  StateMap sm[5];                          // one probability map per hashed model
  int32_t w[256 * 16][8];                  // mixer weights per (node, o1 class)
  uint16_t apm[256][33];                   // SSE transfer per node, 32 buckets

  void Init() {
    const DirectBit dinit;
    std::fill(o1, o1 + 256 * 256, dinit);
    memset(o2, 0, sizeof(o2));
    memset(o3, 0, sizeof(o3));
    memset(o4, 0, sizeof(o4));
    memset(o6, 0, sizeof(o6));
    memset(word, 0, sizeof(word));
    std::fill(pred, pred + 257 * 256, dinit);
    std::fill(o0, o0 + 256, dinit);
    for (int k = 0; k < 5; ++k) sm[k].Init();
    for (int n = 0; n < 256 * 16; ++n) {
      for (int k = 0; k < 8; ++k) w[n][k] = 1 << 14;
    }
    // Identity transfer: bucket centers map to their own squashed value.
    for (int b = 0; b <= 32; ++b) {
      int st = b * 128 - 2048;
      if (st < -kStClamp) st = -kStClamp;
      if (st > kStClamp) st = kStClamp;
      const uint16_t v = SquashTab()[st + kStClamp + 1];
      for (int n = 0; n < 256; ++n) apm[n][b] = v;
    }
  }

  ALWAYS_INLINE static uint32_t Hash2(uint32_t p1, uint32_t p2) {
    return ((p1 | (p2 << 8)) * 2654435761u) >> (32 - kBinHashBits);
  }
  ALWAYS_INLINE static uint32_t Hash3(uint32_t p1, uint32_t p2, uint32_t p3) {
    return ((p1 | (p2 << 8) | (p3 << 16)) * 2654435761u) >> (32 - kBinHashBits);
  }
  ALWAYS_INLINE static uint32_t Hash4(uint32_t p1, uint32_t p2, uint32_t p3,
                                      uint32_t p4) {
    return ((p1 | (p2 << 8) | (p3 << 16) | (p4 << 24)) * 2654435761u) >>
           (32 - kBinHashBits);
  }
  ALWAYS_INLINE static uint32_t Hash6(uint32_t p1, uint32_t p2, uint32_t p3,
                                      uint32_t p4, uint32_t p5, uint32_t p6) {
    const uint32_t lo = p1 | (p2 << 8) | (p3 << 16) | (p4 << 24);
    const uint32_t hi = p5 | (p6 << 8);
    return ((lo * 2654435761u) ^ (hi * 0x9E3779B1u)) >> (32 - kBinHashBits);
  }
  ALWAYS_INLINE static uint32_t Fold(uint32_t h, uint32_t node) {
    return (h ^ (node * 0x1000193u)) & ((1u << kBinHashBits) - 1);
  }
};

ALWAYS_INLINE bool IsWordChar(uint32_t c) {
  const uint32_t l = c | 32u;
  return (l >= 'a' && l <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// Case-folded hash of the alphanumeric run ending just before pos, bounded at
// 24 bytes. A pure function of the shared history window -- recomputed at
// every literal on both sides, so there is no incremental state to keep in
// lockstep and no chunk-seam special case (the scan clamps at the chunk
// start identically on both sides).
ALWAYS_INLINE uint32_t WordHashAt(const uint8_t* buf, size_t pos) {
  size_t n = 0;
  while (n < 24 && pos - n > 0 && IsWordChar(buf[pos - n - 1])) ++n;
  uint32_t h = 0;
  for (size_t j = pos - n; j < pos; ++j) h = h * 0x8B3B15u + (buf[j] | 32u);
  return h;
}

// Per-literal context handles; hashes are computed once per literal and only
// folded with the tree node per bit.
struct BinLitCtx {
  uint32_t o1_base, h2, h3, h4, h6, hw, pred_base, wsel;
  ALWAYS_INLINE static BinLitCtx Make(uint32_t p1, uint32_t p2, uint32_t p3,
                                      uint32_t p4, uint32_t p5, uint32_t p6,
                                      uint32_t whash, uint32_t pred_byte_or_256) {
    BinLitCtx c;
    c.o1_base = p1 << 8;
    c.h2 = BinLitModels::Hash2(p1, p2);
    c.h3 = BinLitModels::Hash3(p1, p2, p3);
    c.h4 = BinLitModels::Hash4(p1, p2, p3, p4);
    c.h6 = BinLitModels::Hash6(p1, p2, p3, p4, p5, p6);
    c.hw = whash;
    c.pred_base = pred_byte_or_256 << 8;
    // Mixer weight-set selector: prediction availability is the strongest
    // regime signal the mixer can specialize on, worth more than one extra
    // bit of the previous byte.
    c.wsel = ((pred_byte_or_256 != 256u ? 1u : 0u) << 3) | ((p1 >> 5) & 7);
    return c;
  }
};

// One probe: state slots, stretched inputs, mixer row, mixed probability, and
// the SSE stage on top. The final probability averages the mixer with the
// learned transfer (1:3 toward the mixer).
struct BinLitProbe {
  DirectBit* d[3];   // o0, o1, pred
  uint8_t* m[5];     // hashed: o2, o3, o4, o6, word
  int32_t st[8];
  int32_t* w;
  uint16_t* apm_slot;
  uint32_t p_mix;
  uint32_t p_final;
};

ALWAYS_INLINE uint32_t BinLitMix(BinLitModels& B, const BinLitCtx& ctx,
                                 uint32_t node, BinLitProbe& pr) {
  const int16_t* stab = StretchTab();
  pr.d[0] = &B.o0[node];
  pr.d[1] = &B.o1[ctx.o1_base | node];
  pr.d[2] = &B.pred[ctx.pred_base | node];
  pr.m[0] = &B.o2[BinLitModels::Fold(ctx.h2, node)];
  pr.m[1] = &B.o3[BinLitModels::Fold(ctx.h3, node)];
  pr.m[2] = &B.o4[BinLitModels::Fold(ctx.h4, node)];
  pr.m[3] = &B.o6[BinLitModels::Fold(ctx.h6, node)];
  pr.m[4] = &B.word[BinLitModels::Fold(ctx.hw, node)];
  int64_t dot = 0;
  pr.w = B.w[(node << 4) | ctx.wsel];
  for (int k = 0; k < 3; ++k) {
    pr.st[k] = stab[pr.d[k]->p];
    dot += static_cast<int64_t>(pr.w[k]) * pr.st[k];
  }
  for (int k = 0; k < 5; ++k) {
    pr.st[3 + k] = stab[B.sm[k].P(*pr.m[k])];
    dot += static_cast<int64_t>(pr.w[3 + k]) * pr.st[3 + k];
  }
  int32_t sm = static_cast<int32_t>(dot >> 16);
  if (sm < -kStClamp) sm = -kStClamp;
  if (sm > kStClamp) sm = kStClamp;
  pr.p_mix = SquashTab()[sm + kStClamp + 1];
  // SSE: piecewise-linear learned transfer over the stretch domain, folded
  // 1:3 toward the mixer -- correction where the models are noisy, neutrality
  // where the mixer is already sharp. (A chained second stage keyed by the
  // mixer's own context measured +/-0.1%: same-key corrective stages are
  // redundant with the weights they sit on.)
  const uint32_t u = static_cast<uint32_t>(sm + 2048);
  const uint32_t b = u >> 7;
  const uint32_t frac = u & 127;
  uint16_t* t = B.apm[node];
  pr.apm_slot = &t[frac >= 64 ? b + 1 : b];
  const uint32_t p_sse = t[b] + (((t[b + 1] - t[b]) * static_cast<int32_t>(frac)) >> 7);
  uint32_t p = (p_sse + 3 * pr.p_mix) >> 2;
  if (p < 1) p = 1;
  if (p > 4095) p = 4095;
  pr.p_final = p;
  return p;
}

ALWAYS_INLINE void BinLitLearn(BinLitModels& B, BinLitProbe& pr, int bit) {
  const uint8_t* nx = StateNext();
  const int32_t err = (bit ? 4095 : 0) - static_cast<int32_t>(pr.p_mix);
  for (int k = 0; k < 8; ++k) {
    int32_t nw = pr.w[k] + ((pr.st[k] * err) >> kBinMixLR);
    if (nw < -kBinWCap) nw = -kBinWCap;
    if (nw > kBinWCap) nw = kBinWCap;
    pr.w[k] = nw;
  }
  for (int k = 0; k < 3; ++k) pr.d[k]->Update(bit);
  for (int k = 0; k < 5; ++k) {
    B.sm[k].Update(*pr.m[k], bit);
    *pr.m[k] = nx[2u * *pr.m[k] + static_cast<unsigned>(bit)];
  }
  const int32_t target = bit ? 4095 : 1;
  *pr.apm_slot = static_cast<uint16_t>(
      *pr.apm_slot + ((target - static_cast<int32_t>(*pr.apm_slot)) >> 6));
}

// ---------------------------------------------------------------------------
// rANS encoder. Symbols are recorded forward (while the models adapt), then a
// segment is encoded in reverse into a scratch buffer.
struct EncEvent {
  uint16_t a;      // cdf: start   | bypass: bits
  uint16_t b;      // cdf: freq    | bypass: 0x8000 | nbits
};

class SegmentEncoder {
public:
  std::vector<EncEvent> events_;

  void Reset() { events_.clear(); }

  ALWAYS_INLINE void PutNib(NibModel& m, size_t sym) {
    events_.push_back(EncEvent{static_cast<uint16_t>(m.Start(sym)),
                               static_cast<uint16_t>(m.Freq(sym))});
    m.Update(sym);
  }

  ALWAYS_INLINE void PutNibBlended(NibModel& a, NibModel& b, uint8_t* w, size_t sym) {
    uint16_t cdf[17];
    BlendCdf(a.cdf_, b.cdf_, *w, cdf);
    events_.push_back(EncEvent{cdf[sym],
                               static_cast<uint16_t>(cdf[sym + 1] - cdf[sym])});
    UpdateBlendWeight(w, a.Freq(sym), b.Freq(sym));
    a.Update(sym);
    b.Update(sym);
  }

  // Three-model cascade: mix(mix(a, b, w1), c, w2). Each stage preserves
  // gap >= min of its inputs' gaps, so strict monotonicity survives the chain.
  ALWAYS_INLINE void PutNibBlended3(NibModel& a, NibModel& b, uint8_t* w1,
                                    NibModel& c, uint8_t* w2, bool use, bool on,
                                    int64_t* gain, size_t sym) {
    uint16_t base[17], mix[17];
    BlendCdf(a.cdf_, b.cdf_, *w1, base);
    const uint16_t* cdf = base;
    if (use) {
      if (c.count_ >= kBlend2Gate) {
        BlendCdf(base, c.cdf_, *w2, mix);
        const uint32_t fbase = static_cast<uint32_t>(base[sym + 1] - base[sym]);
        const uint32_t fmix = static_cast<uint32_t>(mix[sym + 1] - mix[sym]);
        *gain += static_cast<int64_t>(CostTable()[fbase]) - CostTable()[fmix];
        if (on) cdf = mix;
        UpdateBlendWeight2(w2, fbase, c.Freq(sym));
      }
      c.Update(sym);
    }
    events_.push_back(EncEvent{cdf[sym],
                               static_cast<uint16_t>(cdf[sym + 1] - cdf[sym])});
    UpdateBlendWeight(w1, a.Freq(sym), b.Freq(sym));
    a.Update(sym);
    b.Update(sym);
  }

  // Binary symbol with explicit 12-bit probability of bit==1.
  ALWAYS_INLINE void PutBitP(uint32_t p1, int bit) {
    events_.push_back(bit
      ? EncEvent{0, static_cast<uint16_t>(p1)}
      : EncEvent{static_cast<uint16_t>(p1), static_cast<uint16_t>(kProbScale - p1)});
  }

  ALWAYS_INLINE void PutBits(uint32_t bits, uint32_t nbits) {
    while (nbits > 15) {
      PutBits(bits & 0x7FFF, 15);
      bits >>= 15;
      nbits -= 15;
    }
    if (nbits > 0) {
      events_.push_back(EncEvent{static_cast<uint16_t>(bits),
                                 static_cast<uint16_t>(0x8000u | nbits)});
    }
  }

  // Encode all recorded events in reverse; returns bytes appended to out.
  size_t Flush(std::vector<uint8_t>& out) {
    scratch_.resize(events_.size() * 4 + 16);
    uint8_t* base = scratch_.data();
    uint8_t* wp = base + scratch_.size();
    uint32_t x = kRansL;
    for (size_t i = events_.size(); i-- > 0;) {
      const EncEvent e = events_[i];
      if (e.b & 0x8000u) {
        const uint32_t nbits = e.b & 0x7FFFu;
        const uint32_t x_max = (kRansL << 8) >> nbits;
        while (x >= x_max) { *--wp = static_cast<uint8_t>(x); x >>= 8; }
        x = (x << nbits) | e.a;
      } else {
        const uint32_t freq = e.b;
        const uint32_t x_max = ((kRansL >> kProbBits) << 8) * freq;
        while (x >= x_max) { *--wp = static_cast<uint8_t>(x); x >>= 8; }
        x = ((x / freq) << kProbBits) + (x % freq) + e.a;
      }
    }
    for (int i = 0; i < 4; ++i) { *--wp = static_cast<uint8_t>(x); x >>= 8; }
    const size_t n = static_cast<size_t>(base + scratch_.size() - wp);
    out.insert(out.end(), wp, base + scratch_.size());
    events_.clear();
    return n;
  }

private:
  std::vector<uint8_t> scratch_;
};

class SegmentDecoder {
public:
  void Init(const uint8_t* data, size_t len) {
    p_ = data;
    end_ = data + len;
    x_ = 0;
    for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | Get();
  }

  ALWAYS_INLINE size_t GetNib(NibModel& m) {
    const uint32_t slot = x_ & (kProbScale - 1);
    const size_t sym = m.Lookup(slot);
    x_ = m.Freq(sym) * (x_ >> kProbBits) + slot - m.Start(sym);
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    m.Update(sym);
    return sym;
  }

  ALWAYS_INLINE size_t GetNibBlended(NibModel& a, NibModel& b, uint8_t* w) {
    uint16_t cdf[17];
    BlendCdf(a.cdf_, b.cdf_, *w, cdf);
    const uint32_t slot = x_ & (kProbScale - 1);
    const size_t sym = LookupCdf(cdf, slot);
    x_ = (cdf[sym + 1] - cdf[sym]) * (x_ >> kProbBits) + slot - cdf[sym];
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    UpdateBlendWeight(w, a.Freq(sym), b.Freq(sym));
    a.Update(sym);
    b.Update(sym);
    return sym;
  }

  ALWAYS_INLINE size_t GetNibBlended3(NibModel& a, NibModel& b, uint8_t* w1,
                                      NibModel& c, uint8_t* w2, bool use, bool on,
                                      int64_t* gain) {
    uint16_t base[17], mix[17];
    BlendCdf(a.cdf_, b.cdf_, *w1, base);
    const uint16_t* cdf = base;
    const bool hot = use && c.count_ >= kBlend2Gate;
    if (hot && on) {
      BlendCdf(base, c.cdf_, *w2, mix);
      cdf = mix;
    }
    const uint32_t slot = x_ & (kProbScale - 1);
    const size_t sym = LookupCdf(cdf, slot);
    x_ = (cdf[sym + 1] - cdf[sym]) * (x_ >> kProbBits) + slot - cdf[sym];
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    UpdateBlendWeight(w1, a.Freq(sym), b.Freq(sym));
    if (hot) {
      if (!on) BlendCdf(base, c.cdf_, *w2, mix);
      const uint32_t fbase = static_cast<uint32_t>(base[sym + 1] - base[sym]);
      const uint32_t fmix = static_cast<uint32_t>(mix[sym + 1] - mix[sym]);
      *gain += static_cast<int64_t>(CostTable()[fbase]) - CostTable()[fmix];
      UpdateBlendWeight2(w2, fbase, c.Freq(sym));
    }
    a.Update(sym);
    b.Update(sym);
    if (use) c.Update(sym);
    return sym;
  }

  ALWAYS_INLINE int GetBitP(uint32_t p1) {
    const uint32_t slot = x_ & (kProbScale - 1);
    const int bit = slot < p1;
    x_ = (bit ? p1 : kProbScale - p1) * (x_ >> kProbBits) + slot - (bit ? 0 : p1);
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    return bit;
  }

  ALWAYS_INLINE uint32_t GetBits(uint32_t nbits) {
    uint32_t bits = 0, got = 0;
    while (nbits > 15) {
      bits |= GetBitsSmall(15) << got;
      got += 15;
      nbits -= 15;
    }
    if (nbits > 0) bits |= GetBitsSmall(nbits) << got;
    return bits;
  }

private:
  ALWAYS_INLINE uint32_t GetBitsSmall(uint32_t nbits) {
    const uint32_t bits = x_ & ((1u << nbits) - 1);
    x_ >>= nbits;
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    return bits;
  }
  ALWAYS_INLINE uint32_t Get() { return p_ < end_ ? *p_++ : 0; }

  const uint8_t* p_ = nullptr;
  const uint8_t* end_ = nullptr;
  uint32_t x_ = 0;
};

// ---------------------------------------------------------------------------
// Model set shared by encoder and decoder; all updates are driven by decoded
// symbols so both sides stay in lockstep.
//
// Literal contexts (o2-lite): the high nibble sees the previous byte (or, right
// after a match, the byte the match would have continued with) plus the top
// nibble of the byte before that; the low nibble sees the same primary byte
// plus the decoded high nibble.
struct Models {
  // Width of the second-order literal context (bits of prev2 seen by the
  // high-nibble model). Full order-2 (8 bits) measured WORSE on every corpus:
  // 131K unmixed contexts see ~40 samples each and never warm up. Sparse
  // high-order contexts only pay when blended with a dense low-order model,
  // which is exactly the CM mixing advantage -- the planned P2b literal
  // blend (o1 + hashed-o2 with adaptive per-bucket weight) follows from this.
  static const uint32_t kPrev2Bits = 4;
  // ctx: (LZP candidate present at this position, prev kind, kind before
  // that). The availability bit is known to both sides before the token nibble
  // (the decoder probes the LZP table anyway for literals), and a verified o4
  // candidate is a strong prior for LZP/MATCH over LIT.
  NibModel tok[2 * kTokKinds * kTokKinds];

  ALWAYS_INLINE static size_t TokCtx(bool avail, uint32_t prev_kind,
                                     uint32_t prev_kind2) {
    return (avail ? kTokKinds * kTokKinds : 0) + prev_kind * kTokKinds + prev_kind2;
  }
  NibModel lit_hi[512u << kPrev2Bits];   // (prev|256+match_byte, prev2 bits)
  NibModel lit_lo[512 * 16];             // (prev|256+match_byte, hi)
  NibModel len_lo[5];                    // 0: match, 1: rep0, 2: rep1/2, 3: after-lit rep0, 4: lzp
  NibModel len_mid_hi[5];                // slot-14 lengths: value/16 (0..3)
  NibModel len_mid_lo[5][4];             // slot-14 lengths: value%16
  NibModel dist_hi[2];                   // ctx: len >= 8
  NibModel dist_lo[2][4];                // ctx: (len >= 8, hi)
  NibModel dist_align[2];                // low 4 bits of large distances

  // High-order companions blended into the literal models: an o3 stage and,
  // cascaded on top, an o4 stage with its own tables. (An o4 *context
  // extension* of the o3 stage measured worse -- it diluted the o3 buckets.
  // A third *model* with separate tables is additive: cold o4 buckets keep
  // their weight low and merely shadow the o3 mix instead of replacing it.)
  NibModel blend_hi[1u << kBlendBits];
  NibModel blend_lo[1u << kBlendBits];
  uint8_t bw_hi[1u << kBlendBits];
  uint8_t bw_lo[1u << kBlendBits];
  NibModel blend2_hi[1u << kBlendBits];
  NibModel blend2_lo[1u << kBlendBits];
  uint8_t bw2_hi[1u << kBlendBits];
  uint8_t bw2_lo[1u << kBlendBits];

  // Windowed o4 ledger: both sides accumulate (cost without o4) minus (cost
  // with o4) in 1/16-bit units and mute the o4 stage for the next window when
  // the net is negative. The window is a fixed literal count -- decision
  // latency, not a stream unit, so it deliberately does NOT align with rANS
  // segments: 16K literals bounds the worst pre-mute tax at roughly a tenth
  // of a percent per corpus. A muted window still keeps the ledger (and the
  // weights/tables warm), so the stage revives the moment the data starts
  // rewarding it. Lives in Models so the raw-store escape and the CM-section
  // snapshot roll it back with everything else.
  static const uint32_t kO4WindowMask = 16384 - 1;
  int64_t o4_gain_ = 0;
  uint32_t o4_tick_ = 0;
  bool o4_on_ = true;

  // Binary-mixed literal tier state (used only when the chunk opts in).
  BinLitModels bin;

  Models() {
    memset(bw_hi, kBlendInitW, sizeof(bw_hi));
    memset(bw_lo, kBlendInitW, sizeof(bw_lo));
    memset(bw2_hi, kBlend2InitW, sizeof(bw2_hi));
    memset(bw2_lo, kBlend2InitW, sizeof(bw2_lo));
    bin.Init();
  }

  // Hysteresis: staying on needs a non-negative ledger, but reviving from
  // mute demands 64 bits of measured shadow benefit -- on data whose o4 gain
  // oscillates around zero (machine code), symmetric flipping re-enables the
  // stage for every other window and pays its loss each time.
  ALWAYS_INLINE void O4Tick() {
    if ((++o4_tick_ & kO4WindowMask) == 0) {
      o4_on_ = o4_gain_ >= (o4_on_ ? 0 : 16 * 64);
      o4_gain_ = 0;
    }
  }

  ALWAYS_INLINE static size_t O3Hash(uint32_t p1, uint32_t p2, uint32_t p3) {
    const uint32_t v = p1 | (p2 << 8) | (p3 << 16);
    return (v * 2654435761u) >> (32 - kBlendBits);
  }
  ALWAYS_INLINE static size_t O4Hash(uint32_t p1, uint32_t p2, uint32_t p3,
                                     uint32_t p4) {
    const uint32_t v = p1 | (p2 << 8) | (p3 << 16) | (p4 << 24);
    return (v * 2654435761u) >> (32 - kBlendBits);
  }
  ALWAYS_INLINE static size_t LoBucket(size_t h, uint32_t hi) {
    return (h ^ (hi * 0x9E3Bu)) & ((1u << kBlendBits) - 1);
  }

  ALWAYS_INLINE static size_t LitPrimary(uint32_t prev, bool post_match, uint32_t match_byte) {
    return post_match ? 256u + match_byte : prev;
  }
  ALWAYS_INLINE static size_t LitHiCtx(size_t primary, uint32_t prev2) {
    return (primary << kPrev2Bits) | (prev2 >> (8 - kPrev2Bits));
  }
  ALWAYS_INLINE static size_t LitLoCtx(size_t primary, uint32_t hi) {
    return primary * 16u + hi;
  }
  ALWAYS_INLINE static size_t LenCtx(uint32_t kind, uint32_t prev_kind) {
    if (kind == kTokMatch) return 0;
    if (kind == kTokLzp) return 4;
    if (kind == kTokRep0) return prev_kind == kTokLit ? 3 : 1;
    return 2;
  }
};

// Bit-cost estimation against the live adaptive models, in 1/16 bit units.
// This is what lets the encoder make rate-aware parse decisions while the
// decoder stays a dumb executor (plan section P2: encoder-side cost oracle).
inline const uint16_t* CostTable() {
  static std::vector<uint16_t> tab;
  if (tab.empty()) {
    tab.resize(kProbScale + 1);
    tab[0] = 16 * 20;
    for (size_t f = 1; f <= kProbScale; ++f) {
      tab[f] = static_cast<uint16_t>(0.5 + -16.0 * std::log2(double(f) / kProbScale));
    }
  }
  return tab.data();
}

ALWAYS_INLINE uint32_t CostNib(const NibModel& m, size_t sym) {
  return CostTable()[m.Freq(sym)];
}

// ---------------------------------------------------------------------------
// Tier-2 literal codec: 8 mixed binary decisions, MSB first. Encoder, decoder
// and price oracle walk the identical probe/mix; only the bit source and the
// presence of updates differ. The effective predicted byte folds the LZP
// candidate and the post-match continuation byte into the same slot -- the
// two sources are exclusive by construction and both mean "the byte a
// content-addressed model expects next".
ALWAYS_INLINE void BinLitEncode(SegmentEncoder& enc, Models& models,
                                const BinLitCtx& ctx, uint32_t c) {
  uint32_t node = 1;
  for (int i = 7; i >= 0; --i) {
    BinLitProbe pr;
    const uint32_t p1 = BinLitMix(models.bin, ctx, node, pr);
    const int bit = (c >> i) & 1;
    enc.PutBitP(p1, bit);
    BinLitLearn(models.bin, pr, bit);
    node = (node << 1) | bit;
  }
}

ALWAYS_INLINE uint32_t BinLitDecode(SegmentDecoder& dec, Models& models,
                                    const BinLitCtx& ctx) {
  uint32_t node = 1;
  for (int i = 0; i < 8; ++i) {
    BinLitProbe pr;
    const uint32_t p1 = BinLitMix(models.bin, ctx, node, pr);
    const int bit = dec.GetBitP(p1);
    BinLitLearn(models.bin, pr, bit);
    node = (node << 1) | bit;
  }
  return node & 0xFF;
}

// Price of coding byte c here, in 1/16-bit units, against the live models
// (no updates -- same freshness contract as CostNibBlended3).
ALWAYS_INLINE uint32_t BinLitCost(Models& models, const BinLitCtx& ctx,
                                  uint32_t c) {
  const uint16_t* cost = CostTable();
  uint32_t total = 0;
  uint32_t node = 1;
  for (int i = 7; i >= 0; --i) {
    BinLitProbe pr;
    const uint32_t p1 = BinLitMix(models.bin, ctx, node, pr);
    const int bit = (c >> i) & 1;
    total += cost[bit ? p1 : kProbScale - p1];
    node = (node << 1) | bit;
  }
  return total;
}

// Distance slotting, LZMA style: 1..3 direct, otherwise slot encodes the top
// two bits and the rest goes out as raw bits.
ALWAYS_INLINE uint32_t DistSlot(uint32_t dist) {
  if (dist < 4) return dist - 1;
  const uint32_t nb = 31 - __builtin_clz(dist);
  return 2 * nb + ((dist >> (nb - 1)) & 1);
}
ALWAYS_INLINE uint32_t DistExtraBits(uint32_t slot) {
  return slot < 4 ? 0 : slot / 2 - 1;
}
ALWAYS_INLINE uint32_t DistBase(uint32_t slot) {
  if (slot < 4) return slot + 1;
  const uint32_t nb = slot / 2;
  return (2 | (slot & 1)) << (nb - 1);
}

// ---------------------------------------------------------------------------
// Binary-tree (bt4) match finder over one chunk. Each hash-4 bucket holds a
// binary search tree of positions ordered by suffix. One walk does both jobs:
// it collects the (len, dist) pareto set -- lengths strictly increasing, each
// with the nearest distance achieving it, so the DP can price every sub-length
// with its own distance -- and re-roots the tree at the current position.
// Work per position is bounded by the depth budget and the nice-length splice,
// which removes the hash chain's pathological cost on repetitive data.
class Bt4MatchFinder {
public:
  static const uint32_t kHashBits = 18;
#ifdef F2_BT_DEPTH
  static const uint32_t kDepth = F2_BT_DEPTH;
#else
  static const uint32_t kDepth = 64;
#endif
  static const uint32_t kSkipDepth = 16;  // reduced budget inside long matches
  static const size_t kNiceLen = 128;     // long enough: splice and stop
  static const size_t kMaxPairs = 64;

  struct Pair { uint32_t len, dist; };

  void Reset(const uint8_t* data, size_t n) {
    data_ = data;
    n_ = n;
    head_.assign(size_t(1) << kHashBits, kNil);
    // son_ needs no clearing: every slot reachable from a head_ entry was
    // written during this chunk's walks (both cut styles terminate the tree).
    son_.resize(2 * n);
  }

  ALWAYS_INLINE size_t GetMatches(size_t pos, size_t max_len, Pair* pairs) {
    return Walk(pos, max_len, pairs, kDepth);
  }
  ALWAYS_INLINE void Insert(size_t pos) {
    Walk(pos, kNiceLen, nullptr, kSkipDepth);
  }

private:
  // LZMA-style bt insertion. pairs == nullptr means insert-only.
  size_t Walk(size_t pos, size_t max_len, Pair* pairs, uint32_t depth) {
    if (pos + 4 > n_) return 0;
    const size_t cap = std::min(max_len, n_ - pos);
    const size_t nice = std::min(cap, kNiceLen);
    const uint32_t h = Hash(pos);
    uint32_t cur = head_[h];
    head_[h] = static_cast<uint32_t>(pos);
    uint32_t* ptr0 = &son_[2 * pos + 1];
    uint32_t* ptr1 = &son_[2 * pos];
    size_t len0 = 0, len1 = 0;
    size_t npairs = 0, best = kMinMatch - 1;
    const uint8_t* a = data_ + pos;
    while (cur != kNil && depth-- != 0) {
      const uint8_t* b = data_ + cur;
      size_t len = std::min(len0, len1);
      while (len < cap && a[len] == b[len]) ++len;
      if (pairs != nullptr && len > best && npairs < kMaxPairs) {
        best = len;
        pairs[npairs].len = static_cast<uint32_t>(len);
        pairs[npairs].dist = static_cast<uint32_t>(pos - cur);
        ++npairs;
      }
      if (len >= nice) {
        // Long enough (or window-capped): adopt cur's subtrees and stop.
        *ptr1 = son_[2 * cur];
        *ptr0 = son_[2 * cur + 1];
        return npairs;
      }
      uint32_t* pair_slot = &son_[2 * cur];
      if (b[len] < a[len]) {
        *ptr1 = cur;
        ptr1 = pair_slot + 1;
        cur = *ptr1;
        len1 = len;
      } else {
        *ptr0 = cur;
        ptr0 = pair_slot;
        cur = *ptr0;
        len0 = len;
      }
    }
    *ptr0 = kNil;
    *ptr1 = kNil;
    return npairs;
  }

  ALWAYS_INLINE uint32_t Hash(size_t pos) const {
    uint32_t v;
    memcpy(&v, data_ + pos, 4);
    return (v * 2654435761u) >> (32 - kHashBits);
  }

  enum : uint32_t { kNil = 0xFFFFFFFFu };
  const uint8_t* data_ = nullptr;
  size_t n_ = 0;
  std::vector<uint32_t> head_;
  std::vector<uint32_t> son_;
};

// ---------------------------------------------------------------------------
// Length coding helpers: 0..13 direct in the slot nibble, 14..77 through two
// modeled nibbles (lengths cluster hard, raw bits waste ~3 of the 6), the
// long tail through 16 raw bits.
template <typename Sink>
ALWAYS_INLINE void EncodeLen(Sink& enc, Models& models, size_t lctx, size_t len,
                             size_t min_len) {
  const size_t v = len - min_len;
  if (v <= kMaxLenDirect) {
    enc.PutNib(models.len_lo[lctx], v);
  } else if (v <= kMaxLenDirect + 1 + 63) {
    enc.PutNib(models.len_lo[lctx], kLenBypass6);
    const uint32_t m = static_cast<uint32_t>(v - (kMaxLenDirect + 1));
    enc.PutNib(models.len_mid_hi[lctx], m >> 4);
    enc.PutNib(models.len_mid_lo[lctx][m >> 4], m & 15);
  } else {
    enc.PutNib(models.len_lo[lctx], kLenBypass16);
    enc.PutBits(static_cast<uint32_t>(v - (kMaxLenDirect + 1 + 64)), 16);
  }
}

ALWAYS_INLINE size_t DecodeLen(SegmentDecoder& dec, Models& models, size_t lctx,
                               size_t min_len) {
  const size_t slot = dec.GetNib(models.len_lo[lctx]);
  if (slot <= kMaxLenDirect) return min_len + slot;
  if (slot == kLenBypass6) {
    const size_t hi = dec.GetNib(models.len_mid_hi[lctx]) & 3;
    const size_t lo = dec.GetNib(models.len_mid_lo[lctx][hi]);
    return min_len + kMaxLenDirect + 1 + ((hi << 4) | lo);
  }
  return min_len + kMaxLenDirect + 1 + 64 + dec.GetBits(16);
}

// ---------------------------------------------------------------------------
// Chunk encoder: single-pass rate-aware parse. Tokens are chosen by comparing
// approximate bit costs against the live adaptive models (the models advance
// as tokens are emitted, so every decision sees current statistics), with a
// one-step lazy check, a 3-deep rep list, and zstd-style acceleration through
// incompressible stretches.
class F2Encoder {
public:
  void CompressChunk(const uint8_t* in, size_t n, std::vector<uint8_t>& out,
                     Models& models, uint32_t& prev1_io, uint32_t& prev2_io,
                     uint32_t& prev3_io, uint32_t& prev4_io) {
    mf_.Reset(in, n);
    lzp_.Reset();
    // The raw-store escape below must also discard this chunk's model updates,
    // or the decoder (which skips a stored chunk entirely) falls out of sync.
    models_backup_.resize(1);
    models_backup_[0] = models;

    std::vector<std::pair<size_t, size_t>> seg_meta;
    std::vector<uint8_t> data;
    size_t pos = 0;
    size_t reps[kNumReps] = {1, 2, 3};
    uint32_t prev1 = prev1_io, prev2 = prev2_io, prev3 = prev3_io, prev4 = prev4_io;
    uint32_t prev_kind = kTokLit, prev_kind2 = kTokLit;
    bool post_match = false;
    size_t seg_tokens = 0;
    enc_.Reset();

    const uint16_t* cost_tab = CostTable();
    (void)cost_tab;

    lzp_ptr_ = 0;
    // ---- Optimal parse: price-frozen DP over short spans, then emission with
    // live model updates. Prices are refreshed every span, so the staleness
    // window is ~kOptSpan tokens; the DP itself never touches the models, so
    // this is a pure encoder-side upgrade with zero decode cost.
    static const size_t kOptSpan = 512;
    static const size_t kFastLen = 64;  // candidates this long truncate the span
    struct Par { uint32_t cost; uint32_t dist; uint16_t len; uint8_t kind; };
    std::vector<Par> dp(kOptSpan + kFastLen + 1);
    std::vector<Par> toks;
    while (pos < n) {
      const size_t span = std::min(n - pos, kOptSpan);
      const size_t dp_cap = std::min(n - pos, kOptSpan + kFastLen);
      for (size_t j = 0; j <= dp_cap; ++j) dp[j].cost = 0xFFFFFFFFu;
      dp[0].cost = 0;
      pred_.assign(span, kNoPred);
      pred_base_ = pos;
      // Two price rows from the span-entry models: availability varies per
      // position (selected by pred_[i] in the scan), prev kinds are frozen at
      // span entry like before.
      uint32_t tok_price[2][kTokKinds];
      for (uint32_t a = 0; a < 2; ++a) {
        const NibModel& tokm = models.tok[Models::TokCtx(a != 0, prev_kind, prev_kind2)];
        for (uint32_t k = 0; k < kTokKinds; ++k) tok_price[a][k] = CostNib(tokm, k);
      }
      const size_t entry_reps[kNumReps] = {reps[0], reps[1], reps[2]};

      auto relax = [&](size_t j, uint32_t cost, uint32_t kind, size_t len, size_t dist) {
        if (cost < dp[j].cost) {
          dp[j] = Par{cost, static_cast<uint32_t>(dist), static_cast<uint16_t>(len),
                      static_cast<uint8_t>(kind)};
        }
      };

      size_t scan_end = span;
      size_t brk_kind = 0, brk_len = 0, brk_dist = 0;
      bool has_break = false;
      for (size_t i = 0; i < span && !has_break; ++i) {
        const size_t abs = pos + i;
        for (; lzp_ptr_ < abs; ++lzp_ptr_) lzp_.Insert(in, lzp_ptr_);
        const uint32_t base_cost = dp[i].cost;
        const size_t cand_cap = std::min(n - abs, kMaxMatchLen + kMinMatch);

        // LZP candidate first: its next byte doubles as the literal
        // prediction context. The table state here (inserts < abs) is exactly
        // the decoder's state at this position, so the recorded prediction is
        // what the decoder will recompute.
        const uint32_t lzp_cand = lzp_.Candidate(in, abs);
        pred_[i] = lzp_cand != kLzpNil ? in[lzp_cand] : kNoPred;
        const uint32_t av = lzp_cand != kLzpNil ? 1 : 0;
        // Literal edge (exact bytes; the post-match fallback context is
        // approximated as absent while pricing).
        {
          const uint32_t p1 = abs >= 1 ? in[abs - 1] : (i == 0 ? prev1 : 0);
          const uint32_t p2 = abs >= 2 ? in[abs - 2] : (i == 0 ? prev2 : 0);
          const uint32_t p3 = abs >= 3 ? in[abs - 3] : (i == 0 ? prev3 : 0);
          const uint32_t p4 = abs >= 4 ? in[abs - 4] : (i == 0 ? prev4 : 0);
          const uint32_t c0 = in[abs];
          uint32_t lc = tok_price[av][kTokLit];
          if (kLitTier == 2) {
            // Post-match continuation is approximated as absent while pricing
            // (matching the tier-0/1 convention above).
            const uint32_t pb = pred_[i] != kNoPred ? pred_[i] : 256u;
            // Bytes 5-6 back fall to 0 inside the first bytes of a chunk;
            // emission and decode use the identical rule, so the seam stays
            // lockstep (affects at most 6 literals per 16MB chunk).
            const uint32_t p5 = abs >= 5 ? in[abs - 5] : 0;
            const uint32_t p6 = abs >= 6 ? in[abs - 6] : 0;
            lc += BinLitCost(models,
                             BinLitCtx::Make(p1, p2, p3, p4, p5, p6,
                                             WordHashAt(in, abs), pb), c0);
          } else {
            const size_t primary = pred_[i] != kNoPred ? 256u + pred_[i] : (p1 & 0xFF);
            const size_t o3h = Models::O3Hash(p1, p2, p3);
            const size_t o4h = kUseO4 ? Models::O4Hash(p1, p2, p3, p4) : 0;
            const size_t lob4 = kUseO4 ? Models::LoBucket(o4h, c0 >> 4) : 0;
            lc +=
              CostNibBlended3(models.lit_hi[Models::LitHiCtx(primary, p2)],
                              models.blend_hi[o3h], models.bw_hi[o3h],
                              models.blend2_hi[o4h], models.bw2_hi[o4h],
                              models.o4_on_, c0 >> 4) +
              CostNibBlended3(models.lit_lo[Models::LitLoCtx(primary, c0 >> 4)],
                              models.blend_lo[Models::LoBucket(o3h, c0 >> 4)],
                              models.bw_lo[Models::LoBucket(o3h, c0 >> 4)],
                              models.blend2_lo[lob4], models.bw2_lo[lob4],
                              models.o4_on_, c0 & 15);
          }
          relax(i + 1, base_cost + lc + kSpeedBias * 4, kTokLit, 1, 0);
        }
        // Rep candidates against the span-entry rep list; emission re-maps or
        // falls back to an explicit match if the live list drifted, so the
        // recorded distance is always what the bytes were verified against.
        for (size_t r = 0; r < kNumReps; ++r) {
          const size_t d = entry_reps[r];
          if (d > abs) continue;
          const uint8_t* a = in + abs;
          const uint8_t* b = a - d;
          if (a[0] != b[0]) continue;
          size_t len = 1;
          const size_t cap = std::min(cand_cap, kFastLen);
          while (len < cap && a[len] == b[len]) ++len;
          if (len >= kFastLen) {
            while (len < cand_cap && a[len] == b[len]) ++len;
            if (!has_break || len > brk_len) {
              has_break = true;
              brk_kind = kTokRep0 + r;
              brk_len = len;
              brk_dist = d;
              scan_end = i;
            }
            continue;
          }
          const size_t lctx = r == 0 ? 1 : 2;
          for (size_t l = kMinRep; l <= len; ++l) {
            relax(i + l, base_cost + tok_price[av][kTokRep0 + r] + kSpeedBias * 3 +
                  LenCost(models, lctx, l, kMinRep), kTokRep0 + r, l, d);
          }
        }
        // LZP copy-token candidates (same candidate as above).
        {
          const uint32_t cand = lzp_cand;
          if (cand != kLzpNil) {
            const uint8_t* a = in + abs;
            const uint8_t* b = in + cand;
            size_t len = 0;
            const size_t cap = std::min(cand_cap, kFastLen);
            while (len < cap && a[len] == b[len]) ++len;
            if (len >= kFastLen) {
              while (len < cand_cap && a[len] == b[len]) ++len;
              if (!has_break || len > brk_len) {
                has_break = true;
                brk_kind = kTokLzp;
                brk_len = len;
                brk_dist = abs - cand;
                scan_end = i;
              }
            } else if (len >= kMinLzpLen) {
              for (size_t l = kMinLzpLen; l <= len; ++l) {
                relax(i + l, base_cost + tok_price[av][kTokLzp] + kSpeedBias * 3 +
                      LenCost(models, 4, l, kMinLzpLen), kTokLzp, l, abs - cand);
              }
            }
          }
        }
        // Explicit matches: the bt4 (len, dist) pareto set, so every
        // sub-length is priced with its own nearest distance and its own
        // distance context (the old single-candidate form priced all
        // sub-lengths with the longest match's distance).
        {
          Bt4MatchFinder::Pair pairs[Bt4MatchFinder::kMaxPairs];
          const size_t np = mf_.GetMatches(abs, cand_cap, pairs);
          if (np != 0 && pairs[np - 1].len >= kFastLen) {
            if (!has_break || pairs[np - 1].len > brk_len) {
              has_break = true;
              brk_kind = kTokMatch;
              brk_len = pairs[np - 1].len;
              brk_dist = pairs[np - 1].dist;
              scan_end = i;
            }
          } else {
            size_t lo = kMinMatch;
            const uint32_t mtok = tok_price[av][kTokMatch] + kSpeedBias * 5;
            for (size_t k = 0; k < np; ++k) {
              const size_t plen = pairs[k].len;
              const size_t pdist = pairs[k].dist;
              if (plen < lo) continue;
              const uint32_t dp_short = DistCost(models, 0, pdist);
              const uint32_t dp_long = plen >= 8 ? DistCost(models, 1, pdist) : 0;
              for (size_t l = lo; l <= plen; ++l) {
                relax(i + l, base_cost + mtok + LenCost(models, 0, l, kMinMatch) +
                      (l >= 8 ? dp_long : dp_short), kTokMatch, l, pdist);
              }
              lo = plen + 1;
            }
          }
        }
      }

      // Choose the end node and backtrack the token list.
      size_t end_j;
      if (has_break) {
        end_j = scan_end;
      } else {
        end_j = span;
        for (size_t j = span; j <= dp_cap; ++j) {
          if (dp[j].cost < dp[end_j].cost) end_j = j;
        }
      }
      toks.clear();
      for (size_t j = end_j; j > 0;) {
        toks.push_back(dp[j]);
        j -= dp[j].len;
      }

      // Emit backwards-collected tokens forward, with live model updates.
      for (size_t t = toks.size(); t-- > 0;) {
        EmitParsed(models, in, n, toks[t].kind, toks[t].len, toks[t].dist,
                   pos, reps, prev1, prev2, prev3, prev4, post_match,
                   prev_kind, prev_kind2, seg_tokens, seg_meta, data);
      }
      if (has_break) {
        InsertSpan(pos + 1, brk_len - 1);
        EmitParsed(models, in, n, static_cast<uint32_t>(brk_kind), brk_len, brk_dist,
                   pos, reps, prev1, prev2, prev3, prev4, post_match,
                   prev_kind, prev_kind2, seg_tokens, seg_meta, data);
      }
    }
    if (seg_tokens != 0) FlushSegment(models, seg_meta, data, seg_tokens);

    std::vector<uint8_t> body;
    WriteLeb(body, n);
    body.push_back(0);  // mode 0: entropy-coded
    body.push_back(static_cast<uint8_t>(kLitTier));  // literal coder tier
    WriteLeb(body, seg_meta.size());
    size_t off = 0;
    for (const auto& sm : seg_meta) {
      WriteLeb(body, sm.first);
      WriteLeb(body, sm.second);
      body.insert(body.end(), data.data() + off, data.data() + off + sm.second);
      off += sm.second;
    }
    // Store escape for incompressible chunks (with model rollback).
    if (body.size() >= n + n / 128 + 8) {
      models = models_backup_[0];
      body.clear();
      WriteLeb(body, n);
      body.push_back(1);  // mode 1: raw
      body.insert(body.end(), in, in + n);
      prev1 = in[n - 1];
      prev2 = n >= 2 ? in[n - 2] : prev1;
      prev3 = n >= 3 ? in[n - 3] : prev2;
      prev4 = n >= 4 ? in[n - 4] : prev3;
    }
    WriteLeb(out, body.size());
    out.insert(out.end(), body.begin(), body.end());
    prev1_io = prev1;
    prev2_io = prev2;
    prev3_io = prev3;
    prev4_io = prev4;
  }

  static void WriteLeb(std::vector<uint8_t>& out, size_t v) {
    while (v >= 0x80) {
      out.push_back(static_cast<uint8_t>(v) | 0x80);
      v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
  }

private:
  ALWAYS_INLINE void EmitTok(Models& models, uint32_t& prev_kind, uint32_t& prev_kind2,
                             uint32_t kind, bool avail) {
    enc_.PutNib(models.tok[Models::TokCtx(avail, prev_kind, prev_kind2)], kind);
    prev_kind2 = prev_kind;
    prev_kind = kind;
  }

  // Predicted byte at pos (kNoPred if none), exactly as the decoder computes
  // it. In-span positions read the scan-time snapshot; positions beyond the
  // scanned span (a short rep crossing the span end got demoted to literals)
  // have no entry, so catch the insert pointer up and query live -- at that
  // moment the table state is exactly the decoder's (inserts [0, pos)). Using
  // kNoPred there instead desynchronizes the contexts and corrupts the stream.
  ALWAYS_INLINE uint32_t PredAt(const uint8_t* in, size_t pos) {
    const size_t rel = pos - pred_base_;
    if (rel < pred_.size()) return pred_[rel];
    for (; lzp_ptr_ < pos; ++lzp_ptr_) lzp_.Insert(in, lzp_ptr_);
    const uint32_t cand = lzp_.Candidate(in, pos);
    return cand != kLzpNil ? in[cand] : kNoPred;
  }

  ALWAYS_INLINE void EmitLit(Models& models, const uint8_t* in, size_t& pos,
                             const size_t* reps, uint32_t& prev1, uint32_t& prev2,
                             uint32_t& prev3, uint32_t& prev4, bool& post_match,
                             uint32_t pred) {
    const uint32_t c = in[pos];
    const uint32_t hi = c >> 4;
    size_t primary;
    if (pred != kNoPred) {
      primary = 256u + pred;
    } else {
      const uint32_t match_byte = post_match ? in[pos - reps[0]] : 0;
      primary = Models::LitPrimary(prev1, post_match, match_byte);
    }
    const size_t o3h = Models::O3Hash(prev1, prev2, prev3);
    const size_t o4h = kUseO4 ? Models::O4Hash(prev1, prev2, prev3, prev4) : 0;
#ifdef F2_TRACE
    fprintf(F2TraceFile(0), "L %zu pri=%zu o3=%zu o4=%zu p1=%u p2=%u p3=%u pm=%d r0=%zu\n",
            pos, primary, o3h, o4h, prev1, prev2, prev3, (int)post_match, reps[0]);
#endif
    if (kLitTier == 2) {
      const uint32_t pb = primary >= 256u ? static_cast<uint32_t>(primary - 256u) : 256u;
      const uint32_t p5 = pos >= 5 ? in[pos - 5] : 0;
      const uint32_t p6 = pos >= 6 ? in[pos - 6] : 0;
      BinLitEncode(enc_, models,
                   BinLitCtx::Make(prev1, prev2, prev3, prev4, p5, p6,
                                   WordHashAt(in, pos), pb), c);
    } else {
      enc_.PutNibBlended3(models.lit_hi[Models::LitHiCtx(primary, prev2)],
                          models.blend_hi[o3h], &models.bw_hi[o3h],
                          models.blend2_hi[o4h], &models.bw2_hi[o4h],
                          kUseO4, models.o4_on_, &models.o4_gain_, hi);
      const size_t lob = Models::LoBucket(o3h, hi);
      const size_t lob4 = kUseO4 ? Models::LoBucket(o4h, hi) : 0;
      enc_.PutNibBlended3(models.lit_lo[Models::LitLoCtx(primary, hi)],
                          models.blend_lo[lob], &models.bw_lo[lob],
                          models.blend2_lo[lob4], &models.bw2_lo[lob4],
                          kUseO4, models.o4_on_, &models.o4_gain_, c & 15);
      if (kUseO4) models.O4Tick();
    }
    prev4 = prev3;
    prev3 = prev2;
    prev2 = prev1;
    prev1 = c;
    ++pos;
    post_match = false;
  }

  ALWAYS_INLINE void EmitDist(Models& models, size_t len, size_t dist) {
    const uint32_t slot = DistSlot(static_cast<uint32_t>(dist));
    const size_t dctx = len >= 8 ? 1 : 0;
    enc_.PutNib(models.dist_hi[dctx], slot >> 4);
    enc_.PutNib(models.dist_lo[dctx][std::min<uint32_t>(slot >> 4, 3)], slot & 15);
    const uint32_t ebits = DistExtraBits(slot);
    const uint32_t extra = static_cast<uint32_t>(dist) - DistBase(slot);
    if (ebits > 4) {
      enc_.PutBits(extra >> 4, ebits - 4);
      enc_.PutNib(models.dist_align[dctx], extra & 15);
    } else if (ebits > 0) {
      enc_.PutBits(extra, ebits);
    }
  }

  ALWAYS_INLINE static uint32_t CostNibBlended(const NibModel& a, const NibModel& b,
                                               uint32_t w, size_t sym) {
    const int32_t a0 = a.cdf_[sym], a1 = a.cdf_[sym + 1];
    const int32_t b0 = b.cdf_[sym], b1 = b.cdf_[sym + 1];
    const int32_t c0 = sym == 0 ? 0 : a0 + (((b0 - a0) * static_cast<int32_t>(w)) >> 8);
    const int32_t c1 = sym == 15 ? static_cast<int32_t>(kProbScale)
                                 : a1 + (((b1 - a1) * static_cast<int32_t>(w)) >> 8);
    return CostTable()[static_cast<uint32_t>(c1 - c0)];
  }

  ALWAYS_INLINE static uint32_t CostNibBlended3(const NibModel& a, const NibModel& b,
                                                uint32_t w1, const NibModel& c,
                                                uint32_t w2, bool on, size_t sym) {
    const int32_t a0 = a.cdf_[sym], a1 = a.cdf_[sym + 1];
    const int32_t b0 = b.cdf_[sym], b1 = b.cdf_[sym + 1];
    const int32_t m0 = sym == 0 ? 0 : a0 + (((b0 - a0) * static_cast<int32_t>(w1)) >> 8);
    const int32_t m1 = sym == 15 ? static_cast<int32_t>(kProbScale)
                                 : a1 + (((b1 - a1) * static_cast<int32_t>(w1)) >> 8);
    if (!kUseO4 || !on || c.count_ < kBlend2Gate) {
      return CostTable()[static_cast<uint32_t>(m1 - m0)];
    }
    const int32_t d0 = c.cdf_[sym], d1 = c.cdf_[sym + 1];
    const int32_t f0 = sym == 0 ? 0 : m0 + (((d0 - m0) * static_cast<int32_t>(w2)) >> 8);
    const int32_t f1 = sym == 15 ? static_cast<int32_t>(kProbScale)
                                 : m1 + (((d1 - m1) * static_cast<int32_t>(w2)) >> 8);
    return CostTable()[static_cast<uint32_t>(f1 - f0)];
  }

  ALWAYS_INLINE static uint32_t LenCost(const Models& models, size_t lctx,
                                        size_t len, size_t min_len) {
    const NibModel& m = models.len_lo[lctx];
    const size_t v = len - min_len;
    if (v <= kMaxLenDirect) return CostNib(m, v);
    if (v <= kMaxLenDirect + 1 + 63) {
      const uint32_t mid = static_cast<uint32_t>(v - (kMaxLenDirect + 1));
      return CostNib(m, kLenBypass6) +
        CostNib(models.len_mid_hi[lctx], mid >> 4) +
        CostNib(models.len_mid_lo[lctx][mid >> 4], mid & 15);
    }
    return CostNib(m, kLenBypass16) + 16 * 16;
  }

  ALWAYS_INLINE static uint32_t DistCost(const Models& models, size_t dctx, size_t dist) {
    const uint32_t slot = DistSlot(static_cast<uint32_t>(dist));
    const uint32_t ebits = DistExtraBits(slot);
    uint32_t cost = CostNib(models.dist_hi[dctx], slot >> 4) +
      CostNib(models.dist_lo[dctx][std::min<uint32_t>(slot >> 4, 3)], slot & 15);
    if (ebits > 4) {
      cost += (ebits - 4) * 16 +
        CostNib(models.dist_align[dctx],
                (static_cast<uint32_t>(dist) - DistBase(slot)) & 15);
    } else {
      cost += ebits * 16;
    }
    return cost;
  }

  // Emission for DP-parsed tokens. Rep tokens re-map against the live rep list
  // (the DP priced them against the span-entry list); if the distance is no
  // longer on the list it degrades to an explicit match, and a too-short rep
  // degrades to literals -- the recorded distance is what the bytes were
  // verified against, so every degradation stays byte-exact.
  void EmitParsed(Models& models, const uint8_t* in, size_t n, uint32_t kind,
                  size_t len, size_t dist, size_t& pos, size_t* reps,
                  uint32_t& prev1, uint32_t& prev2, uint32_t& prev3,
                  uint32_t& prev4, bool& post_match, uint32_t& prev_kind,
                  uint32_t& prev_kind2, size_t& seg_tokens,
                  std::vector<std::pair<size_t, size_t>>& seg_meta,
                  std::vector<uint8_t>& data) {
    (void)n;
    const uint32_t pred0 = PredAt(in, pos);
    const bool avail0 = pred0 != kNoPred;
    if (kind == kTokLit) {
      EmitTok(models, prev_kind, prev_kind2, kTokLit, avail0);
      EmitLit(models, in, pos, reps, prev1, prev2, prev3, prev4, post_match, pred0);
    } else {
      if (kind >= kTokRep0 && kind <= kTokRep2) {
        size_t r_found = kNumReps;
        for (size_t r = 0; r < kNumReps; ++r) {
          if (reps[r] == dist) { r_found = r; break; }
        }
        if (r_found < kNumReps) {
          kind = static_cast<uint32_t>(kTokRep0 + r_found);
        } else if (len >= kMinMatch) {
          kind = kTokMatch;
        } else {
          for (size_t k = 0; k < len; ++k) {
            const uint32_t p = k == 0 ? pred0 : PredAt(in, pos);
            EmitTok(models, prev_kind, prev_kind2, kTokLit, p != kNoPred);
            EmitLit(models, in, pos, reps, prev1, prev2, prev3, prev4, post_match, p);
            ++seg_tokens;
            if (seg_tokens == kSegTokens) FlushSegment(models, seg_meta, data, seg_tokens);
          }
          return;
        }
      }
      if (kind == kTokMatch) {
        EmitTok(models, prev_kind, prev_kind2, kTokMatch, avail0);
        EncodeLen(enc_, models, 0, len, kMinMatch);
        EmitDist(models, len, dist);
        for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
        reps[0] = dist;
      } else if (kind == kTokLzp) {
        EmitTok(models, prev_kind, prev_kind2, kTokLzp, avail0);
        EncodeLen(enc_, models, 4, len, kMinLzpLen);
        if (dist != reps[0]) {
          for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
          reps[0] = dist;
        }
      } else {
        EmitTok(models, prev_kind, prev_kind2, kind, avail0);
        EncodeLen(enc_, models, Models::LenCtx(kind, prev_kind2), len, kMinRep);
        const size_t r_idx = kind - kTokRep0;
        const size_t d = reps[r_idx];
        for (size_t r = r_idx; r > 0; --r) reps[r] = reps[r - 1];
        reps[0] = d;
      }
#ifdef F2_TRACE
      fprintf(F2TraceFile(0), "T %zu k=%u len=%zu dist=%zu r0=%zu r1=%zu r2=%zu\n",
              pos, kind, len, dist, reps[0], reps[1], reps[2]);
#endif
      pos += len;
      prev1 = in[pos - 1];
      prev2 = pos >= 2 ? in[pos - 2] : prev1;
      prev3 = pos >= 3 ? in[pos - 3] : prev2;
      prev4 = pos >= 4 ? in[pos - 4] : prev3;
      post_match = true;
    }
    ++seg_tokens;
    if (seg_tokens == kSegTokens) FlushSegment(models, seg_meta, data, seg_tokens);
  }

  void FlushSegment(Models& models,
                    std::vector<std::pair<size_t, size_t>>& seg_meta,
                    std::vector<uint8_t>& data, size_t& seg_tokens) {
    (void)models;
    const size_t bytes = enc_.Flush(data);
    seg_meta.push_back({seg_tokens, bytes});
    seg_tokens = 0;
  }

  ALWAYS_INLINE void InsertSpan(size_t from, size_t count) {
    // Inside very long matches a sparse sample of positions is enough.
    const size_t end = from + count;
    if (count <= 128) {
      for (size_t i = from; i < end; ++i) mf_.Insert(i);
    } else {
      for (size_t i = from; i < from + 64; ++i) mf_.Insert(i);
      for (size_t i = from + 64; i < end; i += 4) mf_.Insert(i);
    }
  }

  Bt4MatchFinder mf_;
  SegmentEncoder enc_;
  LzpTable lzp_;
  std::vector<Models> models_backup_;
  std::vector<uint16_t> pred_;
  size_t pred_base_ = 0;
  size_t lzp_ptr_ = 0;
};

// ---------------------------------------------------------------------------
class F2Decoder {
public:
  // Decodes one chunk payload (past the payload_len field). Writes raw_len
  // bytes to out and returns raw_len, or 0 on corrupt input.
  size_t DecompressChunk(const uint8_t* in, size_t in_len, uint8_t* out,
                         size_t out_cap, Models& models,
                         uint32_t& prev1_io, uint32_t& prev2_io, uint32_t& prev3_io,
                         uint32_t& prev4_io) {
    const uint8_t* p = in;
    const uint8_t* in_end = in + in_len;
    const size_t raw_len = ReadLeb(p, in_end);
    if (p >= in_end || raw_len > out_cap) return 0;
    const uint8_t mode = *p++;
    if (mode == 1) {
      if (p + raw_len > in_end) return 0;
      memcpy(out, p, raw_len);
      prev1_io = raw_len ? out[raw_len - 1] : prev1_io;
      prev2_io = raw_len >= 2 ? out[raw_len - 2] : prev1_io;
      prev3_io = raw_len >= 3 ? out[raw_len - 3] : prev2_io;
      prev4_io = raw_len >= 4 ? out[raw_len - 4] : prev3_io;
      return raw_len;
    }
    if (p >= in_end) return 0;
    const uint8_t lit_tier = *p++;
    if (lit_tier > 2) return 0;  // corrupt
    const bool use_o4 = lit_tier == 1;
    const size_t n_segments = ReadLeb(p, in_end);
    lzp_.Reset();
    size_t lzp_ptr = 0;
    size_t pos = 0;
    size_t reps[kNumReps] = {1, 2, 3};
    uint32_t prev1 = prev1_io, prev2 = prev2_io, prev3 = prev3_io, prev4 = prev4_io;
    uint32_t prev_kind = kTokLit, prev_kind2 = kTokLit;
    bool post_match = false;
    for (size_t s = 0; s < n_segments; ++s) {
      size_t n_tokens = ReadLeb(p, in_end);
      const size_t data_len = ReadLeb(p, in_end);
      if (p + data_len > in_end) return 0;
      dec_.Init(p, data_len);
      p += data_len;
      while (n_tokens-- != 0) {
        for (; lzp_ptr < pos; ++lzp_ptr) lzp_.Insert(out, lzp_ptr);
        const uint32_t lzp_cand = lzp_.Candidate(out, pos);
        const size_t kind = dec_.GetNib(models.tok[Models::TokCtx(
            lzp_cand != kLzpNil, prev_kind, prev_kind2)]);
        if (kind == kTokLit) {
          if (pos >= raw_len) return 0;
          size_t primary;
          if (lzp_cand != kLzpNil) {
            primary = 256u + out[lzp_cand];
          } else {
            const uint32_t match_byte = post_match ? out[pos - reps[0]] : 0;
            primary = Models::LitPrimary(prev1, post_match, match_byte);
          }
          const size_t o3h = Models::O3Hash(prev1, prev2, prev3);
          const size_t o4h =
            use_o4 ? Models::O4Hash(prev1, prev2, prev3, prev4) : 0;
#ifdef F2_TRACE
          fprintf(F2TraceFile(1), "L %zu pri=%zu o3=%zu o4=%zu p1=%u p2=%u p3=%u pm=%d r0=%zu\n",
                  pos, primary, o3h, o4h, prev1, prev2, prev3, (int)post_match, reps[0]);
#endif
          uint8_t c;
          if (lit_tier == 2) {
            const uint32_t pb =
              primary >= 256u ? static_cast<uint32_t>(primary - 256u) : 256u;
            const uint32_t p5 = pos >= 5 ? out[pos - 5] : 0;
            const uint32_t p6 = pos >= 6 ? out[pos - 6] : 0;
            c = static_cast<uint8_t>(BinLitDecode(
              dec_, models,
              BinLitCtx::Make(prev1, prev2, prev3, prev4, p5, p6,
                              WordHashAt(out, pos), pb)));
          } else {
            const size_t hi = dec_.GetNibBlended3(
              models.lit_hi[Models::LitHiCtx(primary, prev2)],
              models.blend_hi[o3h], &models.bw_hi[o3h],
              models.blend2_hi[o4h], &models.bw2_hi[o4h],
              use_o4, models.o4_on_, &models.o4_gain_);
            const size_t lob = Models::LoBucket(o3h, hi);
            const size_t lob4 = use_o4 ? Models::LoBucket(o4h, hi) : 0;
            const size_t lo = dec_.GetNibBlended3(
              models.lit_lo[Models::LitLoCtx(primary, hi)],
              models.blend_lo[lob], &models.bw_lo[lob],
              models.blend2_lo[lob4], &models.bw2_lo[lob4],
              use_o4, models.o4_on_, &models.o4_gain_);
            if (use_o4) models.O4Tick();
            c = static_cast<uint8_t>((hi << 4) | lo);
          }
          out[pos++] = c;
          prev4 = prev3;
          prev3 = prev2;
          prev2 = prev1;
          prev1 = c;
          post_match = false;
        } else if (kind < kTokKinds) {
          size_t len, dist;
          if (kind == kTokMatch) {
            len = DecodeLen(dec_, models, 0, kMinMatch);
            const size_t dctx = len >= 8 ? 1 : 0;
            const uint32_t hi = static_cast<uint32_t>(dec_.GetNib(models.dist_hi[dctx]));
            const uint32_t lo = static_cast<uint32_t>(
              dec_.GetNib(models.dist_lo[dctx][hi < 3 ? hi : 3]));
            const uint32_t slot = (hi << 4) | lo;
            const uint32_t ebits = DistExtraBits(slot);
            if (ebits > 4) {
              const uint32_t high = dec_.GetBits(ebits - 4);
              const uint32_t low = static_cast<uint32_t>(dec_.GetNib(models.dist_align[dctx]));
              dist = DistBase(slot) + ((high << 4) | low);
            } else {
              dist = DistBase(slot) + (ebits > 0 ? dec_.GetBits(ebits) : 0);
            }
            for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
            reps[0] = dist;
          } else if (kind == kTokLzp) {
            len = DecodeLen(dec_, models, 4, kMinLzpLen);
            const uint32_t cand = lzp_cand;
            if (cand == kLzpNil) return 0;  // corrupt: encoder had a candidate
            dist = pos - cand;
            if (dist != reps[0]) {
              for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
              reps[0] = dist;
            }
          } else {
            len = DecodeLen(dec_, models, Models::LenCtx(
              static_cast<uint32_t>(kind), prev_kind), kMinRep);
            const size_t r_idx = kind - kTokRep0;
            dist = reps[r_idx];
            for (size_t r = r_idx; r > 0; --r) reps[r] = reps[r - 1];
            reps[0] = dist;
          }
          if (dist == 0 || dist > pos || len > raw_len - pos) return 0;
#ifdef F2_TRACE
          fprintf(F2TraceFile(1), "T %zu k=%zu len=%zu dist=%zu r0=%zu r1=%zu r2=%zu\n",
                  pos, kind, len, dist, reps[0], reps[1], reps[2]);
#endif
          CopyMatch(out, pos, dist, len);
          pos += len;
          prev1 = out[pos - 1];
          prev2 = pos >= 2 ? out[pos - 2] : prev1;
          prev3 = pos >= 3 ? out[pos - 3] : prev2;
          prev4 = pos >= 4 ? out[pos - 4] : prev3;
          post_match = true;
        } else {
          return 0;  // corrupt
        }
        prev_kind2 = prev_kind;
        prev_kind = static_cast<uint32_t>(kind);
      }
    }
    if (pos != raw_len) return 0;
    prev1_io = prev1;
    prev2_io = prev2;
    prev3_io = prev3;
    prev4_io = prev4;
    return raw_len;
  }

  static size_t ReadLeb(const uint8_t*& p, const uint8_t* end) {
    size_t v = 0;
    int shift = 0;
    while (p < end) {
      const uint8_t b = *p++;
      v |= static_cast<size_t>(b & 0x7F) << shift;
      if ((b & 0x80) == 0) break;
      shift += 7;
    }
    return v;
  }

private:
  ALWAYS_INLINE static void CopyMatch(uint8_t* out, size_t pos, size_t dist, size_t len) {
    uint8_t* dst = out + pos;
    const uint8_t* src = dst - dist;
    if (dist >= 8) {
      size_t i = 0;
      for (; i + 8 <= len; i += 8) memcpy(dst + i, src + i, 8);
      for (; i < len; ++i) dst[i] = src[i];
    } else {
      for (size_t i = 0; i < len; ++i) dst[i] = src[i];
    }
  }

  SegmentDecoder dec_;
  LzpTable lzp_;
};

// ---------------------------------------------------------------------------
// Memory-level API (also used by the standalone tests/benchmarks).
inline void MemCompress(const uint8_t* in, size_t n, std::vector<uint8_t>& out) {
  F2Encoder enc;
  std::vector<Models> models(1);
  uint32_t prev1 = 0, prev2 = 0, prev3 = 0, prev4 = 0;
  size_t pos = 0;
  while (pos < n) {
    const size_t chunk = std::min(n - pos, kChunkSize);
    enc.CompressChunk(in + pos, chunk, out, models[0], prev1, prev2, prev3, prev4);
    pos += chunk;
  }
}

inline bool MemDecompress(const uint8_t* in, size_t n, uint8_t* out, size_t out_n) {
  F2Decoder dec;
  std::vector<Models> models(1);
  uint32_t prev1 = 0, prev2 = 0, prev3 = 0, prev4 = 0;
  size_t in_pos = 0, out_pos = 0;
  while (out_pos < out_n && in_pos < n) {
    const uint8_t* p = in + in_pos;
    const size_t payload = F2Decoder::ReadLeb(p, in + n);
    const size_t header = static_cast<size_t>(p - (in + in_pos));
    if (payload == 0 || in_pos + header + payload > n) return false;
    const size_t got = dec.DecompressChunk(p, payload, out + out_pos, out_n - out_pos,
                                           models[0], prev1, prev2, prev3, prev4);
    if (got == 0) return false;
    in_pos += header + payload;
    out_pos += got;
  }
  return out_pos == out_n;
}

#ifndef F2_NO_ARCHIVE
// ---------------------------------------------------------------------------
// Stream Compressor wrapper for the Archive container, with the ratio-budget
// selector: every chunk is trial-compressed with both F2 and an embedded CM
// fast-mode engine, and F2 is kept only while the cumulative size overhead
// stays inside kBudgetPermille of the cumulative CM size. This makes the
// compression-ratio bound a construction-level guarantee (worst case is the
// all-CM encoding), while decode speed rises automatically as the F2 gap
// narrows corpus by corpus.
class F2 : public Compressor {
public:
  // The ratio guarantee is enforced at section level against a CONTINUOUS CM
  // reference: a section is emitted as F2 chunks only if their total stays
  // within kGatePermille of the whole-section CM stream, otherwise the
  // section ships as that CM stream itself. A per-chunk budget against a
  // chunked CM reference is NOT sound -- chunking can inflate the reference
  // arbitrarily on data with long-range redundancy (measured +53% on a
  // corpus of three nearly identical trees), which would let F2 through the
  // gate while blowing the real bound.
  enum : uint64_t { kGatePermille = 30 };
  enum : uint8_t { kModeCM = 2 };
  // Sections cap the buffering; losses across 256MB section boundaries are
  // negligible next to the 3% budget.
  static const size_t kSectionSize = 256 * MB;

  F2(const FrequencyCounter<256>& freq, Detector::Profile profile)
    : freq_(freq), profile_(profile) {}

  static uint8_t CmMemForChunk(size_t chunk_size) {
    uint8_t level = 0;
    while (level < 8 && ((uint64_t(2) * MB) << level) < 8 * uint64_t(chunk_size)) {
      ++level;
    }
    return level;
  }

  virtual void compress(Stream* in, Stream* out, uint64_t max_count) OVERRIDE {
    // F2 models and byte context run continuously across sections (mirroring
    // the decoder); a section that ships as CM rolls them back via snapshot.
    F2Encoder enc;
    std::vector<Models> models(1), snapshot(1);
    uint32_t prev1 = 0, prev2 = 0, prev3 = 0, prev4 = 0;
    std::vector<uint8_t> buf;
    while (max_count > 0) {
      const size_t want = static_cast<size_t>(std::min<uint64_t>(max_count, kSectionSize));
      buf.resize(want);
      size_t got = 0;
      while (got < want) {
        const size_t r = in->read(buf.data() + got, want - got);
        if (r == 0) break;
        got += r;
      }
      if (got == 0) break;
      CompressSection(buf.data(), got, out, enc, models, snapshot, prev1, prev2,
                      prev3, prev4);
      max_count -= got;
    }
  }

  void CompressSection(const uint8_t* data, size_t n, Stream* out, F2Encoder& enc,
                       std::vector<Models>& models, std::vector<Models>& snapshot,
                       uint32_t& prev1, uint32_t& prev2, uint32_t& prev3,
                       uint32_t& prev4) {
    // Reference: the section as one continuous CM fast-mode stream.
    std::vector<uint8_t> cm_data;
    const uint8_t cm_mem = CmMemForChunk(n);
    {
      cm::CM<4, false> cm_engine(freq_, cm_mem, true, profile_);
      ReadMemoryStream rms(data, data + n);
      WriteVectorStream wvs(&cm_data);
      cm_engine.compress(&rms, &wvs, n);
    }
    // Candidate: the section as F2 chunks (advances the shared models).
    snapshot[0] = models[0];
    const uint32_t old1 = prev1, old2 = prev2, old3 = prev3;
    (void)old2; (void)old3;
    std::vector<uint8_t> f2_data;
    {
      size_t pos = 0;
      while (pos < n) {
        const size_t chunk = std::min(n - pos, kChunkSize);
        std::vector<uint8_t> payload;
        enc.CompressChunk(data + pos, chunk, payload, models[0], prev1, prev2,
                          prev3, prev4);
        F2Encoder::WriteLeb(f2_data, payload.size());
        f2_data.insert(f2_data.end(), payload.begin(), payload.end());
        pos += chunk;
      }
    }
    const uint64_t gate = uint64_t(cm_data.size()) * (1000 + kGatePermille) / 1000;
    if (uint64_t(f2_data.size()) <= gate) {
      out->write(f2_data.data(), f2_data.size());
    } else {
      // CM section: undo the F2 trial's model advances, set the byte context
      // from the raw tail exactly like the decoder will.
      models[0] = snapshot[0];
      prev1 = n ? data[n - 1] : old1;
      prev2 = n >= 2 ? data[n - 2] : prev1;
      prev3 = n >= 3 ? data[n - 3] : prev2;
      prev4 = n >= 4 ? data[n - 4] : prev3;
      std::vector<uint8_t> payload;
      F2Encoder::WriteLeb(payload, n);
      payload.push_back(kModeCM);
      payload.push_back(cm_mem);
      payload.insert(payload.end(), cm_data.begin(), cm_data.end());
      std::vector<uint8_t> hdr;
      F2Encoder::WriteLeb(hdr, payload.size());
      out->write(hdr.data(), hdr.size());
      out->write(payload.data(), payload.size());
    }
  }

  virtual void decompress(Stream* in, Stream* out, uint64_t max_count) OVERRIDE {
    F2Decoder dec;
    std::vector<Models> models(1);
    uint32_t prev1 = 0, prev2 = 0, prev3 = 0, prev4 = 0;
    std::vector<uint8_t> comp;
    std::vector<uint8_t> raw(64 * KB);
    while (max_count > 0) {
      size_t payload = 0;
      int shift = 0;
      for (;;) {
        const int c = in->get();
        if (c < 0) return;
        payload |= static_cast<size_t>(c & 0x7F) << shift;
        if ((c & 0x80) == 0) break;
        shift += 7;
      }
      if (payload == 0) return;
      comp.resize(payload);
      if (in->read(comp.data(), payload) != payload) return;
      const uint8_t* peek = comp.data();
      const uint8_t* peek_end = comp.data() + payload;
      const size_t raw_len = F2Decoder::ReadLeb(peek, peek_end);
      if (peek >= peek_end || raw_len > kSectionSize) return;
      if (raw_len > raw.size()) raw.resize(raw_len);
      size_t got = 0;
      if (*peek == kModeCM) {
        ++peek;
        if (peek >= peek_end) return;
        const uint8_t cm_mem = *peek++;
        cm::CM<4, false> cm_engine(freq_, cm_mem, true, profile_);
        ReadMemoryStream rms(peek, peek_end);
        WriteVectorStream wvs(&cm_scratch_);
        cm_scratch_.clear();
        cm_engine.decompress(&rms, &wvs, raw_len);
        if (cm_scratch_.size() != raw_len) return;
        memcpy(raw.data(), cm_scratch_.data(), raw_len);
        got = raw_len;
        prev1 = raw_len ? raw[raw_len - 1] : prev1;
        prev2 = raw_len >= 2 ? raw[raw_len - 2] : prev1;
        prev3 = raw_len >= 3 ? raw[raw_len - 3] : prev2;
        prev4 = raw_len >= 4 ? raw[raw_len - 4] : prev3;
      } else {
        got = dec.DecompressChunk(comp.data(), payload, raw.data(), raw.size(),
                                  models[0], prev1, prev2, prev3, prev4);
        if (got == 0) return;
      }
      out->write(raw.data(), got);
      max_count -= std::min<uint64_t>(max_count, got);
    }
  }

private:
  FrequencyCounter<256> freq_;
  Detector::Profile profile_;
  std::vector<uint8_t> scratch_;
  std::vector<uint8_t> cm_scratch_;
};
#endif  // F2_NO_ARCHIVE

}  // namespace f2

#endif  // _F2_HPP_
