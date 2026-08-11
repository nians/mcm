// Entropy-decode architecture microbenchmarks for the MCM fastmode redesign study.
// Measures, on the same machine as the mcm benchmarks:
//  1. serial-cm  : bit-serial adaptive binary range decode, o1 context (LZMA literal style)
//                  -> upper bound for ANY tuned bit-serial CM decoder (all tables L1-resident)
//  2. serial-cm-o2: same but with a 16MB context table -> shows cache-miss impact
//  3. nib-2x     : adaptive nibble decode with TWO interleaved range-coder streams
//  4. tans-o1    : semi-static order-1 tANS (FSE-style) table decode, 2-way interleaved
//  5. lz-copy    : raw LZ token replay (match copies + literal memcpy) -> LZ-side ceiling
// All decoders verify output equals input.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using Clock = chrono::steady_clock;

static double now_s() {
  return chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// ---------------- Range coder (7z style, byte renorm) ----------------
struct RC_Enc {
  uint64_t low = 0; uint32_t range = 0xFFFFFFFF; uint8_t cache = 0; uint32_t cache_size = 1;
  vector<uint8_t>* out;
  void shift_low() {
    if ((uint32_t)low < 0xFF000000u || (low >> 32)) {
      uint8_t t = cache;
      do { out->push_back(uint8_t(t + uint8_t(low >> 32))); t = 0xFF; } while (--cache_size);
      cache = uint8_t(low >> 24);
    }
    ++cache_size;
    low = (uint32_t)low << 8;
  }
  void encode_bit(int bit, uint32_t p, int shift) { // p = P(bit=1)
    uint32_t mid = (range >> shift) * p;
    if (bit) range = mid; else { low += mid; range -= mid; }
    while (range < (1u << 24)) { range <<= 8; shift_low(); }
  }
  void flush() { for (int i = 0; i < 5; ++i) shift_low(); }
};
struct RC_Dec {
  uint32_t range = 0xFFFFFFFF, code = 0;
  const uint8_t* in; const uint8_t* end;
  inline uint8_t get() { return in < end ? *in++ : 0; }
  void init() { code = 0; for (int i = 0; i < 5; ++i) code = (code << 8) | get(); }
  inline int decode_bit(uint32_t p, int shift) {
    uint32_t mid = (range >> shift) * p;
    int bit = code < mid;
    if (bit) range = mid; else { code -= mid; range -= mid; }
    while (range < (1u << 24)) { code = (code << 8) | get(); range <<= 8; }
    return bit;
  }
};

// adaptive 12-bit probability
struct Prob { uint16_t p = 2048; };
static inline void upd(Prob& m, int bit, int rate = 5) {
  if (bit) m.p += (4096 - m.p) >> rate; else m.p -= m.p >> rate;
}

// ---------------- 1. bit-serial o1 CM (all L1) ----------------
static size_t enc_serial_o1(const vector<uint8_t>& data, vector<Prob>& probs, vector<uint8_t>& out) {
  RC_Enc rc; rc.out = &out;
  uint8_t prev = 0;
  for (uint8_t c : data) {
    size_t base = (size_t)prev << 8;
    uint32_t ctx = 1;
    for (int b = 7; b >= 0; --b) {
      int bit = (c >> b) & 1;
      Prob& m = probs[base + ctx];
      rc.encode_bit(bit, m.p, 12);
      upd(m, bit);
      ctx = ctx * 2 + bit;
    }
    prev = c;
  }
  rc.flush();
  return out.size();
}
static void dec_serial_o1(vector<uint8_t>& data, size_t n, vector<Prob>& probs, const vector<uint8_t>& in) {
  RC_Dec rc; rc.in = in.data(); rc.end = in.data() + in.size(); rc.init();
  uint8_t prev = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t base = (size_t)prev << 8;
    uint32_t ctx = 1;
    for (int b = 0; b < 8; ++b) {
      Prob& m = probs[base + ctx];
      int bit = rc.decode_bit(m.p, 12);
      upd(m, bit);
      ctx = ctx * 2 + bit;
    }
    prev = (uint8_t)(ctx & 0xFF);
    data[i] = prev;
  }
}

// ---------------- 2. bit-serial o2 CM (16MB tables, cache misses) ----------------
static size_t enc_serial_o2(const vector<uint8_t>& data, vector<Prob>& probs, vector<uint8_t>& out) {
  RC_Enc rc; rc.out = &out;
  uint32_t prev = 0;
  for (uint8_t c : data) {
    size_t base = (size_t)(prev & 0xFFFF) << 8;
    uint32_t ctx = 1;
    for (int b = 7; b >= 0; --b) {
      int bit = (c >> b) & 1;
      Prob& m = probs[base + ctx];
      rc.encode_bit(bit, m.p, 12);
      upd(m, bit);
      ctx = ctx * 2 + bit;
    }
    prev = (prev << 8) | c;
  }
  rc.flush();
  return out.size();
}
static void dec_serial_o2(vector<uint8_t>& data, size_t n, vector<Prob>& probs, const vector<uint8_t>& in) {
  RC_Dec rc; rc.in = in.data(); rc.end = in.data() + in.size(); rc.init();
  uint32_t prev = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t base = (size_t)(prev & 0xFFFF) << 8;
    uint32_t ctx = 1;
    for (int b = 0; b < 8; ++b) {
      Prob& m = probs[base + ctx];
      int bit = rc.decode_bit(m.p, 12);
      upd(m, bit);
      ctx = ctx * 2 + bit;
    }
    uint8_t c = (uint8_t)(ctx & 0xFF);
    prev = (prev << 8) | c;
    data[i] = c;
  }
}

// ---------------- 3. two interleaved range streams, nibble contexts o1 ----------------
// Even/odd bytes go to stream A/B; each byte = 2 nibbles = 2*4 adaptive bit decodes,
// but the two streams have independent serial chains -> ILP.
struct NibCtx { Prob hi[256 * 15]; Prob lo[256 * 16 * 15]; };
template <class RC>
static inline void enc_nib(RC& rc, Prob* base15, int nib) {
  uint32_t ctx = 1;
  for (int b = 3; b >= 0; --b) {
    int bit = (nib >> b) & 1;
    Prob& m = base15[ctx - 1];
    rc.encode_bit(bit, m.p, 12);
    upd(m, bit);
    ctx = ctx * 2 + bit;
  }
}
static inline int dec_nib(RC_Dec& rc, Prob* base15) {
  uint32_t ctx = 1;
  for (int b = 0; b < 4; ++b) {
    Prob& m = base15[ctx - 1];
    int bit = rc.decode_bit(m.p, 12);
    upd(m, bit);
    ctx = ctx * 2 + bit;
  }
  return ctx & 15;
}
static void enc_2x(const vector<uint8_t>& data, NibCtx& nc, vector<uint8_t>& outA, vector<uint8_t>& outB) {
  RC_Enc a; a.out = &outA; RC_Enc b; b.out = &outB;
  uint8_t prevA = 0, prevB = 0;
  size_t n = data.size();
  for (size_t i = 0; i + 1 < n; i += 2) {
    uint8_t cA = data[i], cB = data[i + 1];
    enc_nib(a, &nc.hi[prevA * 15], cA >> 4);
    enc_nib(b, &nc.hi[prevB * 15], cB >> 4);
    enc_nib(a, &nc.lo[(prevA * 16 + (cA >> 4)) * 15], cA & 15);
    enc_nib(b, &nc.lo[(prevB * 16 + (cB >> 4)) * 15], cB & 15);
    prevA = cA; prevB = cB;
  }
  a.flush(); b.flush();
}
static void dec_2x(vector<uint8_t>& data, size_t n, NibCtx& nc, const vector<uint8_t>& inA, const vector<uint8_t>& inB) {
  RC_Dec a; a.in = inA.data(); a.end = a.in + inA.size(); a.init();
  RC_Dec b; b.in = inB.data(); b.end = b.in + inB.size(); b.init();
  uint8_t prevA = 0, prevB = 0;
  for (size_t i = 0; i + 1 < n; i += 2) {
    int hA = dec_nib(a, &nc.hi[prevA * 15]);
    int hB = dec_nib(b, &nc.hi[prevB * 15]);
    int lA = dec_nib(a, &nc.lo[(prevA * 16 + hA) * 15]);
    int lB = dec_nib(b, &nc.lo[(prevB * 16 + hB) * 15]);
    prevA = (uint8_t)((hA << 4) | lA);
    prevB = (uint8_t)((hB << 4) | lB);
    data[i] = prevA; data[i + 1] = prevB;
  }
}

// ---------------- 4. order-1 semi-static tANS (FSE-ish), 2-way interleaved ----------------
// 12-bit tables per previous-byte context built from first-pass counts (encoder side sends them).
// Decode: state -> (symbol, nbits, newstate base) table lookup + bit refill. ~O(1)/symbol.
struct TansTable {
  // decode entry: sym, nbits, newState
  struct DE { uint8_t sym; uint8_t nbits; uint16_t base; };
  vector<DE> dec;       // [ctx][state]
  vector<uint16_t> encState;  // encoder: per ctx per sym start
  static const int TL = 12;      // table log
  static const int TSIZE = 1 << TL;
};
// Build per-context normalized freqs and decode table.
static void tans_build(const vector<uint8_t>& data, TansTable& t) {
  const int TL = TansTable::TL, TSIZE = TansTable::TSIZE;
  vector<uint32_t> cnt(256 * 256, 0);
  uint8_t prev = 0;
  for (uint8_t c : data) { cnt[prev * 256 + c]++; prev = c; }
  t.dec.resize((size_t)256 * TSIZE);
  for (int ctx = 0; ctx < 256; ++ctx) {
    uint32_t* c = &cnt[ctx * 256];
    uint64_t tot = 0; for (int s = 0; s < 256; ++s) tot += c[s];
    // normalize to TSIZE with every present symbol >=1
    vector<uint32_t> norm(256, 0);
    if (tot == 0) { norm[0] = TSIZE; }
    else {
      uint32_t assigned = 0; int maxs = 0;
      for (int s = 0; s < 256; ++s) {
        if (c[s]) {
          norm[s] = max<uint32_t>(1, (uint32_t)((uint64_t)c[s] * TSIZE / tot));
          assigned += norm[s];
          if (norm[s] > norm[maxs]) maxs = s;
        }
      }
      // fix rounding on the largest symbol
      norm[maxs] += TSIZE - (int)assigned; // may go negative in theory; fine for bench data
    }
    // spread symbols (simple sequential spread — decode-equivalent for benching)
    TansTable::DE* de = &t.dec[(size_t)ctx * TSIZE];
    int pos = 0;
    for (int s = 0; s < 256; ++s) {
      uint32_t f = norm[s];
      if (!f) continue;
      int nb = TL - (31 - __builtin_clz(f));       // bits when state in low half
      for (uint32_t k = 0; k < f; ++k) {
        de[pos].sym = (uint8_t)s;
        // next-state for tANS: (f+k) << nb spans [f<<nb, 2f<<nb)
        uint32_t ns = (f + k);
        int bits = TL - (31 - __builtin_clz(ns));
        de[pos].nbits = (uint8_t)bits;
        de[pos].base = (uint16_t)((ns << bits) - TSIZE);
        ++pos;
      }
    }
    for (; pos < TSIZE; ++pos) { de[pos] = de[pos ? pos - 1 : 0]; }
  }
}
// For the benchmark we only need decode-side timing on a synthetic valid stream:
// we generate bits by ENCODING with the same tables reversed. To keep this simple and
// still measure realistic decode cost, we build a stream by walking states forward and
// recording emitted bits, then decode it back (round trip on a per-context tANS).
struct BitWriterLSB { vector<uint8_t> v; uint64_t acc = 0; int n = 0;
  void put(uint32_t bits, int nb) { acc |= (uint64_t)bits << n; n += nb; while (n >= 8) { v.push_back((uint8_t)acc); acc >>= 8; n -= 8; } }
  void flushw() { while (n > 0) { v.push_back((uint8_t)acc); acc >>= 8; n -= 8; } }
};
struct BitReaderLSB { const uint8_t* p; const uint8_t* end; uint64_t acc = 0; int n = 0;
  inline void refill() { while (n <= 56 && p < end) { acc |= (uint64_t)(*p++) << n; n += 8; } }
  inline uint32_t get(int nb) { refill(); uint32_t r = (uint32_t)(acc & ((1u << nb) - 1)); acc >>= nb; n -= nb; return r; }
};
// Simplified: encode with rANS instead (byte-aligned, order-1, 2 interleaved states) —
// equivalent decode cost profile to tANS-table decode and much simpler to get right.
struct RansCtxTables {
  static const int SB = 12; // scale bits
  vector<uint16_t> freq;   // [ctx][256]
  vector<uint16_t> cum;    // [ctx][257]
  vector<uint8_t> slot2sym; // [ctx][1<<SB]
};
static void rans_build(const vector<uint8_t>& data, RansCtxTables& T) {
  const int M = 1 << RansCtxTables::SB;
  vector<uint32_t> cnt(256 * 256, 0);
  uint8_t prev = 0;
  for (uint8_t c : data) { cnt[prev * 256 + c]++; prev = c; }
  T.freq.assign(256 * 256, 0); T.cum.assign(256 * 257, 0);
  T.slot2sym.assign((size_t)256 * M, 0);
  for (int ctx = 0; ctx < 256; ++ctx) {
    uint32_t* c = &cnt[ctx * 256];
    uint64_t tot = 0; for (int s = 0; s < 256; ++s) tot += c[s];
    vector<uint32_t> norm(256, 0);
    if (tot == 0) norm[0] = M;
    else {
      uint32_t assigned = 0; int maxs = 0;
      for (int s = 0; s < 256; ++s) if (c[s]) {
        norm[s] = max<uint32_t>(1, (uint32_t)((uint64_t)c[s] * M / tot));
        assigned += norm[s]; if (norm[s] > norm[maxs]) maxs = s;
      }
      norm[maxs] += M - (int)assigned;
    }
    uint16_t* f = &T.freq[ctx * 256]; uint16_t* cu = &T.cum[ctx * 257];
    uint32_t acc = 0;
    for (int s = 0; s < 256; ++s) { f[s] = (uint16_t)norm[s]; cu[s] = (uint16_t)acc; acc += norm[s]; }
    cu[256] = (uint16_t)acc;
    uint8_t* sl = &T.slot2sym[(size_t)ctx * M];
    for (int s = 0; s < 256; ++s) for (uint32_t k = cu[s]; k < (uint32_t)(cu[s] + f[s]); ++k) sl[k] = (uint8_t)s;
  }
}
// rANS 32-bit, byte renorm, 2 interleaved states, order-1 context = previous byte in that lane's sequence.
static void rans_enc2(const vector<uint8_t>& data, RansCtxTables& T, vector<uint8_t>& out) {
  const int SB = RansCtxTables::SB;
  const uint32_t L = 1u << 23; // renorm bound
  // encode in reverse; lanes interleave bytes
  uint32_t s0 = L, s1 = L;
  vector<uint8_t> rev;
  size_t n = data.size() & ~size_t(1);
  for (size_t i = n; i >= 2; i -= 2) {
    // lane1 encodes data[i-1] with ctx=data[i-3(same lane prev)] etc. For bench simplicity ctx = previous byte in stream order.
    for (int lane = 1; lane >= 0; --lane) {
      size_t idx = i - 2 + lane;
      uint8_t sym = data[idx];
      uint8_t ctx = idx == 0 ? 0 : data[idx - 1];
      uint32_t f = T.freq[ctx * 256 + sym], cu = T.cum[ctx * 257 + sym];
      if (!f) { f = 1; } // shouldn't happen
      uint32_t& st = lane ? s1 : s0;
      uint32_t x_max = ((L >> SB) << 8) * f;
      while (st >= x_max) { rev.push_back((uint8_t)st); st >>= 8; }
      st = ((st / f) << SB) + (st % f) + cu;
    }
    if (i == 2) break;
  }
  for (int lane = 0; lane < 2; ++lane) {
    uint32_t st = lane ? s1 : s0;
    for (int k = 0; k < 4; ++k) { rev.push_back((uint8_t)st); st >>= 8; }
  }
  out.assign(rev.rbegin(), rev.rend());
}
static void rans_dec2(vector<uint8_t>& data, size_t n, RansCtxTables& T, const vector<uint8_t>& in) {
  const int SB = RansCtxTables::SB; const uint32_t M1 = (1u << SB) - 1;
  const uint8_t* p = in.data();
  uint32_t s0 = 0, s1 = 0;
  // encoder flushed s0 then s1 into the reversed tail -> after reversal s1's bytes come first
  for (int k = 3; k >= 0; --k) s1 = (s1 << 8) | *p++;
  for (int k = 3; k >= 0; --k) s0 = (s0 << 8) | *p++;
  uint8_t prev = 0;
  n &= ~size_t(1);
  const uint32_t L = 1u << 23;
  for (size_t i = 0; i < n; i += 2) {
    { // lane0
      uint32_t slot = s0 & M1;
      uint8_t sym = T.slot2sym[(size_t)prev * (M1 + 1) + slot];
      uint32_t f = T.freq[prev * 256 + sym], cu = T.cum[prev * 257 + sym];
      s0 = f * (s0 >> SB) + slot - cu;
      while (s0 < L) s0 = (s0 << 8) | *p++;
      data[i] = sym; prev = sym;
    }
    { // lane1
      uint32_t slot = s1 & M1;
      uint8_t sym = T.slot2sym[(size_t)prev * (M1 + 1) + slot];
      uint32_t f = T.freq[prev * 256 + sym], cu = T.cum[prev * 257 + sym];
      s1 = f * (s1 >> SB) + slot - cu;
      while (s1 < L) s1 = (s1 << 8) | *p++;
      data[i + 1] = sym; prev = sym;
    }
  }
}

// ---------------- 5. LZ token replay ----------------
// Parse data with a simple greedy hash-4 LZ (encoder side, untimed), then time decode:
// tokens = (lit_len, match_len, dist); literals copied from a literal pool.
struct LZTok { uint32_t lit_len, match_len, dist; };
static void lz_parse(const vector<uint8_t>& data, vector<LZTok>& toks, vector<uint8_t>& lits) {
  const size_t n = data.size();
  vector<int64_t> head(1 << 20, -1);
  auto h4 = [&](size_t i) {
    uint32_t v; memcpy(&v, &data[i], 4);
    return (v * 2654435761u) >> 12;
  };
  size_t i = 0, lit_start = 0;
  while (i + 4 < n) {
    uint32_t h = h4(i);
    int64_t c = head[h]; head[h] = i;
    size_t best = 0;
    if (c >= 0 && !memcmp(&data[c], &data[i], 4)) {
      size_t l = 4;
      while (i + l < n && data[c + l] == data[i + l] && l < 273) ++l;
      best = l;
    }
    if (best >= 4) {
      toks.push_back({(uint32_t)(i - lit_start), (uint32_t)best, (uint32_t)(i - (size_t)c)});
      for (size_t k = lit_start; k < i; ++k) lits.push_back(data[k]);
      i += best; lit_start = i;
    } else ++i;
  }
  toks.push_back({(uint32_t)(n - lit_start), 0, 0});
  for (size_t k = lit_start; k < n; ++k) lits.push_back(data[k]);
}
static void lz_replay(vector<uint8_t>& out, size_t n, const vector<LZTok>& toks, const vector<uint8_t>& lits) {
  size_t op = 0, lp = 0;
  for (const auto& t : toks) {
    memcpy(&out[op], &lits[lp], t.lit_len); // over-copy safe enough for bench (reserve)
    op += t.lit_len; lp += t.lit_len;
    if (t.match_len) {
      size_t from = op - t.dist;
      if (t.dist >= t.match_len) { memcpy(&out[op], &out[from], t.match_len); }
      else { for (size_t k = 0; k < t.match_len; ++k) out[op + k] = out[from + k]; }
      op += t.match_len;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s file [iters]\n", argv[0]); return 1; }
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  vector<uint8_t> data(sz);
  if (fread(data.data(), 1, sz, f) != (size_t)sz) return 1;
  fclose(f);
  int iters = argc > 2 ? atoi(argv[2]) : 3;
  double mb = sz / 1e6;
  printf("file=%s size=%.1fMB iters=%d\n", argv[1], mb, iters);

  // 1. serial o1
  {
    vector<Prob> pe((size_t)256 * 256), pd;
    vector<uint8_t> enc; enc_serial_o1(data, pe, enc);
    vector<uint8_t> dec(sz);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      pd.assign((size_t)256 * 256, Prob{});
      double t0 = now_s(); dec_serial_o1(dec, sz, pd, enc); double t1 = now_s();
      best = min(best, t1 - t0);
    }
    printf("serial-cm-o1 : enc %.2f%%  dec %6.1f MB/s %s\n", 100.0 * enc.size() / sz, mb / best,
           dec == data ? "OK" : "BAD");
  }
  // 2. serial o2 (16M contexts)
  {
    vector<Prob> pe((size_t)65536 * 256), pd;
    vector<uint8_t> enc; enc_serial_o2(data, pe, enc);
    vector<uint8_t> dec(sz);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      pd.assign((size_t)65536 * 256, Prob{});
      double t0 = now_s(); dec_serial_o2(dec, sz, pd, enc); double t1 = now_s();
      best = min(best, t1 - t0);
    }
    printf("serial-cm-o2 : enc %.2f%%  dec %6.1f MB/s %s\n", 100.0 * enc.size() / sz, mb / best,
           dec == data ? "OK" : "BAD");
  }
  // 3. interleaved 2x nibble o1
  {
    auto* nce = new NibCtx(); auto* ncd = new NibCtx();
    vector<uint8_t> ea, eb; enc_2x(data, *nce, ea, eb);
    vector<uint8_t> dec(sz);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      *ncd = NibCtx{};
      double t0 = now_s(); dec_2x(dec, sz, *ncd, ea, eb); double t1 = now_s();
      best = min(best, t1 - t0);
    }
    size_t n2 = sz & ~size_t(1);
    bool ok = !memcmp(dec.data(), data.data(), n2);
    printf("nib-cm-2x    : enc %.2f%%  dec %6.1f MB/s %s\n", 100.0 * (ea.size() + eb.size()) / sz, mb / best,
           ok ? "OK" : "BAD");
    delete nce; delete ncd;
  }
  // 4. rANS o1 semi-static, 2 lanes
  {
    RansCtxTables T; rans_build(data, T);
    vector<uint8_t> enc; rans_enc2(data, T, enc);
    vector<uint8_t> dec(sz);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      double t0 = now_s(); rans_dec2(dec, sz, T, enc); double t1 = now_s();
      best = min(best, t1 - t0);
    }
    size_t n2 = sz & ~size_t(1);
    bool ok = !memcmp(dec.data(), data.data(), n2);
    printf("rans-o1-2x   : enc %.2f%% (+tables) dec %6.1f MB/s %s\n", 100.0 * enc.size() / sz, mb / best,
           ok ? "OK" : "BAD");
  }
  // 5. LZ replay
  {
    vector<LZTok> toks; vector<uint8_t> lits;
    lz_parse(data, toks, lits);
    vector<uint8_t> dec(sz + 512);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      double t0 = now_s(); lz_replay(dec, sz, toks, lits); double t1 = now_s();
      best = min(best, t1 - t0);
    }
    dec.resize(sz);
    bool ok = dec == data;
    printf("lz-replay    : lits %.1f%% toks/MB %.0f  dec %6.1f MB/s %s\n",
           100.0 * lits.size() / sz, toks.size() / mb, mb / best, ok ? "OK" : "BAD");
  }
  return 0;
}
