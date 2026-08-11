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
#include <vector>

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
static const uint32_t kProbBits = 12;
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
    size_t sym = 0;
    while (cdf_[sym + 1] <= slot) ++sym;
    return sym;
  }

  ALWAYS_INLINE void Update(size_t sym) {
    const int rate = count_ < 16 ? 3 : (count_ < 64 ? 4 : 5);
    count_ += count_ < 64;
    for (int i = 1; i < 16; ++i) {
      const int target = i <= static_cast<int>(sym)
        ? i
        : static_cast<int>(kProbScale) - (16 - i);
      cdf_[i] = static_cast<uint16_t>(cdf_[i] + ((target - static_cast<int>(cdf_[i])) >> rate));
    }
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

ALWAYS_INLINE void BlendCdf(const uint16_t* a, const uint16_t* b, uint32_t w,
                            uint16_t* out) {
  out[0] = 0;
  out[16] = static_cast<uint16_t>(kProbScale);
  for (int i = 1; i < 16; ++i) {
    const int32_t ai = a[i];
    out[i] = static_cast<uint16_t>(ai + (((b[i] - ai) * static_cast<int32_t>(w)) >> 8));
  }
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
    size_t sym = 0;
    while (cdf[sym + 1] <= slot) ++sym;
    x_ = (cdf[sym + 1] - cdf[sym]) * (x_ >> kProbBits) + slot - cdf[sym];
    while (x_ < kRansL) x_ = (x_ << 8) | Get();
    UpdateBlendWeight(w, a.Freq(sym), b.Freq(sym));
    a.Update(sym);
    b.Update(sym);
    return sym;
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
  NibModel tok[kTokKinds * kTokKinds];   // ctx: (prev kind, kind before that)
  NibModel lit_hi[512u << kPrev2Bits];   // (prev|256+match_byte, prev2 bits)
  NibModel lit_lo[512 * 16];             // (prev|256+match_byte, hi)
  NibModel len_lo[5];                    // 0: match, 1: rep0, 2: rep1/2, 3: after-lit rep0, 4: lzp
  NibModel len_mid_hi[5];                // slot-14 lengths: value/16 (0..3)
  NibModel len_mid_lo[5][4];             // slot-14 lengths: value%16
  NibModel dist_hi[2];                   // ctx: len >= 8
  NibModel dist_lo[2][4];                // ctx: (len >= 8, hi)
  NibModel dist_align[2];                // low 4 bits of large distances

  // High-order (o3) companions blended into the literal models.
  NibModel blend_hi[1u << kBlendBits];
  NibModel blend_lo[1u << kBlendBits];
  uint8_t bw_hi[1u << kBlendBits];
  uint8_t bw_lo[1u << kBlendBits];

  Models() {
    memset(bw_hi, kBlendInitW, sizeof(bw_hi));
    memset(bw_lo, kBlendInitW, sizeof(bw_lo));
  }

  ALWAYS_INLINE static size_t O3Hash(uint32_t p1, uint32_t p2, uint32_t p3) {
    const uint32_t v = p1 | (p2 << 8) | (p3 << 16);
    return (v * 2654435761u) >> (32 - kBlendBits);
  }
  // Note: an order-4 companion context and a cold-bucket blend bypass were
  // both measured slightly worse than plain o3 + always-blend; o3 it is.
  ALWAYS_INLINE static size_t LoBucket(size_t o3h, uint32_t hi) {
    return (o3h ^ (hi * 0x9E3Bu)) & ((1u << kBlendBits) - 1);
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
// Hash-chain match finder over one chunk.
class ChainMatchFinder {
public:
  static const uint32_t kHashBits = 18;
  static const uint32_t kMaxChain = 96;
  static const size_t kNiceLen = 128;  // long enough, stop searching

  void Reset(const uint8_t* data, size_t n) {
    data_ = data;
    n_ = n;
    head_.assign(size_t(1) << kHashBits, kNil);
    chain_.resize(n);
  }

  ALWAYS_INLINE void Insert(size_t pos) {
    if (pos + 4 > n_) return;
    const uint32_t h = Hash(pos);
    chain_[pos] = head_[h];
    head_[h] = static_cast<uint32_t>(pos);
  }

  ALWAYS_INLINE size_t Find(size_t pos, size_t max_len, size_t* out_dist) const {
    if (pos + 4 > n_ || max_len < kMinMatch) return 0;
    size_t best_len = 0, best_dist = 0;
    uint32_t cand = head_[Hash(pos)];
    uint32_t depth = kMaxChain;
    const uint8_t* cur = data_ + pos;
    while (cand != kNil && depth-- != 0) {
      const uint8_t* c = data_ + cand;
      if (c[best_len] == cur[best_len]) {
        size_t len = 0;
        while (len < max_len && c[len] == cur[len]) ++len;
        if (len > best_len) {
          best_len = len;
          best_dist = pos - cand;
          if (len >= max_len || len >= kNiceLen) break;
        }
      }
      cand = chain_[cand];
    }
    *out_dist = best_dist;
    return best_len >= kMinMatch ? best_len : 0;
  }

private:
  ALWAYS_INLINE uint32_t Hash(size_t pos) const {
    uint32_t v;
    memcpy(&v, data_ + pos, 4);
    return (v * 2654435761u) >> (32 - kHashBits);
  }

  enum : uint32_t { kNil = 0xFFFFFFFFu };
  const uint8_t* data_ = nullptr;
  size_t n_ = 0;
  std::vector<uint32_t> head_;
  std::vector<uint32_t> chain_;
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
                     uint32_t& prev3_io) {
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
    uint32_t prev1 = prev1_io, prev2 = prev2_io, prev3 = prev3_io;
    uint32_t prev_kind = kTokLit, prev_kind2 = kTokLit;
    bool post_match = false;
    size_t miss_run = 0, next_search = 0;
    size_t seg_tokens = 0;
    enc_.Reset();

    const uint16_t* cost_tab = CostTable();
    (void)cost_tab;

    size_t lzp_ptr = 0;
    while (pos < n) {
      // Publish the positions consumed so far to the LZP table (decoder does
      // exactly the same at the top of its token loop).
      for (; lzp_ptr < pos; ++lzp_ptr) lzp_.Insert(in, lzp_ptr);
      const size_t max_len = std::min(n - pos, kMaxMatchLen + kMinMatch);
      // Candidate: best rep.
      size_t best_rep_len = 0, best_rep = 0;
      for (size_t r = 0; r < kNumReps; ++r) {
        const size_t d = reps[r];
        if (d > pos) continue;
        const uint8_t* a = in + pos;
        const uint8_t* b = a - d;
        if (a[best_rep_len] != b[best_rep_len]) continue;
        size_t len = 0;
        while (len < max_len && a[len] == b[len]) ++len;
        if (len > best_rep_len) {
          best_rep_len = len;
          best_rep = r;
        }
      }
      if (best_rep_len < kMinRep) best_rep_len = 0;
      // Candidate: content-addressed LZP (length-only coding).
      size_t lzp_len = 0, lzp_dist = 0;
      const uint32_t lzp_cand = lzp_.Candidate(in, pos);
      if (lzp_cand != kLzpNil) {
        const uint8_t* a = in + pos;
        const uint8_t* b = in + lzp_cand;
        size_t len = 0;
        while (len < max_len && a[len] == b[len]) ++len;
        if (len >= kMinLzpLen) {
          lzp_len = len;
          lzp_dist = pos - lzp_cand;
        }
      }
      // Candidate: explicit match (throttled; a long rep/lzp wins regardless).
      size_t dist = 0, mlen = 0;
      if (best_rep_len < 32 && lzp_len < 32 && pos >= next_search) {
        mlen = mf_.Find(pos, max_len, &dist);
        if (mlen == 0) next_search = pos + 1 + (miss_run >> 6);
      }

      // Costs (1/16 bit units) against the current models.
      const NibModel& tokm = models.tok[prev_kind * kTokKinds + prev_kind2];
      const uint32_t match_byte = post_match ? in[pos - reps[0]] : 0;
      const size_t lit_primary = Models::LitPrimary(prev1, post_match, match_byte);
      const uint32_t c0 = in[pos];
      const size_t o3h = Models::O3Hash(prev1, prev2, prev3);
      const uint32_t lit_cost = CostNib(tokm, kTokLit) +
        CostNibBlended(models.lit_hi[Models::LitHiCtx(lit_primary, prev2)],
                       models.blend_hi[o3h], models.bw_hi[o3h], c0 >> 4) +
        CostNibBlended(models.lit_lo[Models::LitLoCtx(lit_primary, c0 >> 4)],
                       models.blend_lo[Models::LoBucket(o3h, c0 >> 4)],
                       models.bw_lo[Models::LoBucket(o3h, c0 >> 4)], c0 & 15);

      uint32_t rep_avg = ~0u;
      if (best_rep_len != 0) {
        const uint32_t kind = static_cast<uint32_t>(kTokRep0 + best_rep);
        const uint32_t cost = CostNib(tokm, kind) +
          LenCost(models, Models::LenCtx(kind, prev_kind), best_rep_len, kMinRep);
        rep_avg = cost / static_cast<uint32_t>(best_rep_len);
      }
      uint32_t lzp_avg = ~0u;
      if (lzp_len != 0) {
        const uint32_t cost = CostNib(tokm, kTokLzp) +
          LenCost(models, 4, lzp_len, kMinLzpLen);
        lzp_avg = cost / static_cast<uint32_t>(lzp_len);
      }
      uint32_t match_avg = ~0u;
      if (mlen != 0) {
        match_avg = MatchCost(models, tokm, mlen, dist, prev_kind) /
          static_cast<uint32_t>(mlen);
      }

      const bool lzp_beats_rep = lzp_avg <= rep_avg;
      const uint32_t back_avg = lzp_beats_rep ? lzp_avg : rep_avg;
      const bool rep_beats_match = back_avg <= match_avg;
      const uint32_t seq_avg = rep_beats_match ? back_avg : match_avg;
      if (seq_avg >= lit_cost) {
        // Nothing beats coding one literal.
        mf_.Insert(pos);
        EmitTok(models, prev_kind, prev_kind2, kTokLit);
        EmitLit(models, in, pos, reps, prev1, prev2, prev3, post_match);
        ++seg_tokens;
        ++miss_run;
        if (seg_tokens == kSegTokens) FlushSegment(seg_meta, data, seg_tokens);
        continue;
      }
      miss_run = 0;
      if (!rep_beats_match) {
        // One-step lazy: a longer match at pos+1 can beat this one.
        mf_.Insert(pos);
        if (mlen < 32 && pos + 1 < n) {
          size_t d2 = 0;
          const size_t l2 =
            mf_.Find(pos + 1, std::min(n - pos - 1, kMaxMatchLen + kMinMatch), &d2);
          if (l2 > mlen) {
            const uint32_t next_cost = MatchCost(models, tokm, l2, d2, kTokLit);
            if ((lit_cost + next_cost) / static_cast<uint32_t>(1 + l2) <= match_avg) {
              EmitTok(models, prev_kind, prev_kind2, kTokLit);
              EmitLit(models, in, pos, reps, prev1, prev2, prev3, post_match);
              ++seg_tokens;
              if (seg_tokens == kSegTokens) FlushSegment(seg_meta, data, seg_tokens);
              continue;
            }
          }
        }
        InsertSpan(pos + 1, mlen - 1);
        EmitTok(models, prev_kind, prev_kind2, kTokMatch);
        EncodeLen(enc_, models, 0, mlen, kMinMatch);
        EmitDist(models, mlen, dist);
        for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
        reps[0] = dist;
        pos += mlen;
      } else if (lzp_beats_rep) {
        mf_.Insert(pos);
        InsertSpan(pos + 1, lzp_len - 1);
        EmitTok(models, prev_kind, prev_kind2, kTokLzp);
        EncodeLen(enc_, models, 4, lzp_len, kMinLzpLen);
        if (lzp_dist != reps[0]) {
          for (size_t r = kNumReps; r-- > 1;) reps[r] = reps[r - 1];
          reps[0] = lzp_dist;
        }
        pos += lzp_len;
      } else {
        mf_.Insert(pos);
        InsertSpan(pos + 1, best_rep_len - 1);
        const uint32_t kind = static_cast<uint32_t>(kTokRep0 + best_rep);
        EmitTok(models, prev_kind, prev_kind2, kind);
        EncodeLen(enc_, models, Models::LenCtx(kind, prev_kind2),
                  best_rep_len, kMinRep);
        const size_t d = reps[best_rep];
        for (size_t r = best_rep; r > 0; --r) reps[r] = reps[r - 1];
        reps[0] = d;
        pos += best_rep_len;
      }
      prev1 = in[pos - 1];
      prev2 = pos >= 2 ? in[pos - 2] : prev1;
      prev3 = pos >= 3 ? in[pos - 3] : prev2;
      post_match = true;
      ++seg_tokens;
      if (seg_tokens == kSegTokens) FlushSegment(seg_meta, data, seg_tokens);
    }
    if (seg_tokens != 0) FlushSegment(seg_meta, data, seg_tokens);

    std::vector<uint8_t> body;
    WriteLeb(body, n);
    body.push_back(0);  // mode 0: entropy-coded
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
    }
    WriteLeb(out, body.size());
    out.insert(out.end(), body.begin(), body.end());
    prev1_io = prev1;
    prev2_io = prev2;
    prev3_io = prev3;
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
                             uint32_t kind) {
    enc_.PutNib(models.tok[prev_kind * kTokKinds + prev_kind2], kind);
    prev_kind2 = prev_kind;
    prev_kind = kind;
  }

  ALWAYS_INLINE void EmitLit(Models& models, const uint8_t* in, size_t& pos,
                             const size_t* reps, uint32_t& prev1, uint32_t& prev2,
                             uint32_t& prev3, bool& post_match) {
    const uint32_t c = in[pos];
    const uint32_t match_byte = post_match ? in[pos - reps[0]] : 0;
    const uint32_t hi = c >> 4;
    const size_t primary = Models::LitPrimary(prev1, post_match, match_byte);
    const size_t o3h = Models::O3Hash(prev1, prev2, prev3);
    enc_.PutNibBlended(models.lit_hi[Models::LitHiCtx(primary, prev2)],
                       models.blend_hi[o3h], &models.bw_hi[o3h], hi);
    const size_t lob = Models::LoBucket(o3h, hi);
    enc_.PutNibBlended(models.lit_lo[Models::LitLoCtx(primary, hi)],
                       models.blend_lo[lob], &models.bw_lo[lob], c & 15);
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

  ALWAYS_INLINE static uint32_t MatchCost(const Models& models, const NibModel& tokm,
                                          size_t len, size_t dist, uint32_t prev_kind) {
    const uint32_t slot = DistSlot(static_cast<uint32_t>(dist));
    const size_t dctx = len >= 8 ? 1 : 0;
    const uint32_t ebits = DistExtraBits(slot);
    uint32_t cost = CostNib(tokm, kTokMatch) +
      LenCost(models, 0, len, kMinMatch) +
      CostNib(models.dist_hi[dctx], slot >> 4) +
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

  void FlushSegment(std::vector<std::pair<size_t, size_t>>& seg_meta,
                    std::vector<uint8_t>& data, size_t& seg_tokens) {
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

  ChainMatchFinder mf_;
  SegmentEncoder enc_;
  LzpTable lzp_;
  std::vector<Models> models_backup_;
};

// ---------------------------------------------------------------------------
class F2Decoder {
public:
  // Decodes one chunk payload (past the payload_len field). Writes raw_len
  // bytes to out and returns raw_len, or 0 on corrupt input.
  size_t DecompressChunk(const uint8_t* in, size_t in_len, uint8_t* out,
                         size_t out_cap, Models& models,
                         uint32_t& prev1_io, uint32_t& prev2_io, uint32_t& prev3_io) {
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
      return raw_len;
    }
    const size_t n_segments = ReadLeb(p, in_end);
    lzp_.Reset();
    size_t lzp_ptr = 0;
    size_t pos = 0;
    size_t reps[kNumReps] = {1, 2, 3};
    uint32_t prev1 = prev1_io, prev2 = prev2_io, prev3 = prev3_io;
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
        const size_t kind = dec_.GetNib(models.tok[prev_kind * kTokKinds + prev_kind2]);
        if (kind == kTokLit) {
          if (pos >= raw_len) return 0;
          const uint32_t match_byte = post_match ? out[pos - reps[0]] : 0;
          const size_t primary = Models::LitPrimary(prev1, post_match, match_byte);
          const size_t o3h = Models::O3Hash(prev1, prev2, prev3);
          const size_t hi = dec_.GetNibBlended(
            models.lit_hi[Models::LitHiCtx(primary, prev2)],
            models.blend_hi[o3h], &models.bw_hi[o3h]);
          const size_t lob = Models::LoBucket(o3h, hi);
          const size_t lo = dec_.GetNibBlended(
            models.lit_lo[Models::LitLoCtx(primary, hi)],
            models.blend_lo[lob], &models.bw_lo[lob]);
          const uint8_t c = static_cast<uint8_t>((hi << 4) | lo);
          out[pos++] = c;
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
            const uint32_t cand = lzp_.Candidate(out, pos);
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
          CopyMatch(out, pos, dist, len);
          pos += len;
          prev1 = out[pos - 1];
          prev2 = pos >= 2 ? out[pos - 2] : prev1;
          prev3 = pos >= 3 ? out[pos - 3] : prev2;
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
  uint32_t prev1 = 0, prev2 = 0, prev3 = 0;
  size_t pos = 0;
  while (pos < n) {
    const size_t chunk = std::min(n - pos, kChunkSize);
    enc.CompressChunk(in + pos, chunk, out, models[0], prev1, prev2, prev3);
    pos += chunk;
  }
}

inline bool MemDecompress(const uint8_t* in, size_t n, uint8_t* out, size_t out_n) {
  F2Decoder dec;
  std::vector<Models> models(1);
  uint32_t prev1 = 0, prev2 = 0, prev3 = 0;
  size_t in_pos = 0, out_pos = 0;
  while (out_pos < out_n && in_pos < n) {
    const uint8_t* p = in + in_pos;
    const size_t payload = F2Decoder::ReadLeb(p, in + n);
    const size_t header = static_cast<size_t>(p - (in + in_pos));
    if (payload == 0 || in_pos + header + payload > n) return false;
    const size_t got = dec.DecompressChunk(p, payload, out + out_pos, out_n - out_pos,
                                           models[0], prev1, prev2, prev3);
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
    uint32_t prev1 = 0, prev2 = 0, prev3 = 0;
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
      CompressSection(buf.data(), got, out, enc, models, snapshot, prev1, prev2, prev3);
      max_count -= got;
    }
  }

  void CompressSection(const uint8_t* data, size_t n, Stream* out, F2Encoder& enc,
                       std::vector<Models>& models, std::vector<Models>& snapshot,
                       uint32_t& prev1, uint32_t& prev2, uint32_t& prev3) {
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
        enc.CompressChunk(data + pos, chunk, payload, models[0], prev1, prev2, prev3);
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
    uint32_t prev1 = 0, prev2 = 0, prev3 = 0;
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
      } else {
        got = dec.DecompressChunk(comp.data(), payload, raw.data(), raw.size(),
                                  models[0], prev1, prev2, prev3);
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
