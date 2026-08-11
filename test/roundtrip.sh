#!/bin/bash
# MCM roundtrip correctness matrix.
#
# Usage: test/roundtrip.sh <mcm-binary> <corpus-dir> [work-dir]
#
# For every corpus file and configuration: compress, decompress, verify md5.
# Also checks encoder determinism (two compressions of the same input must
# produce identical archives once the variable metadata prefix is skipped).
# Exits non-zero if any cell fails.
set -u
MCM=$(realpath "$1")
CORPUS_DIR=$(realpath "$2")
WORK=${3:-$(mktemp -d)}
mkdir -p "$WORK"
CONFIGS=("-t8" "-f8" "-m8" "-f6" "-f8 -filter=none" "-f8 -lzp=false")
DECOMP_TIMEOUT=600
fails=0
total=0

for corpus in "$CORPUS_DIR"/*.dat; do
  name=$(basename "$corpus" .dat)
  ref_md5=$(md5sum "$corpus" | cut -d' ' -f1)
  for cfg in "${CONFIGS[@]}"; do
    total=$((total + 1))
    cell="$name $cfg"
    W="$WORK/rt_$(echo "$cell" | tr ' =-' '___')"
    rm -rf "$W" && mkdir -p "$W" && cd "$W" || exit 2
    cp "$corpus" input.dat
    if ! $MCM $cfg input.dat a.mcm >comp.log 2>&1; then
      echo "FAIL $cell (compress error)"; fails=$((fails + 1)); continue
    fi
    rm input.dat
    if ! timeout $DECOMP_TIMEOUT $MCM d a.mcm >decomp.log 2>&1; then
      echo "FAIL $cell (decompress error/timeout)"; fails=$((fails + 1)); continue
    fi
    out_md5=$(md5sum input.dat 2>/dev/null | cut -d' ' -f1)
    if [ "$out_md5" != "$ref_md5" ]; then
      echo "FAIL $cell (verify mismatch)"; fails=$((fails + 1)); continue
    fi
    echo "PASS $cell ($(stat -c%s a.mcm) bytes)"
  done

  # Encoder determinism: metadata carries timestamps, so compare the data
  # tail after the variable-length metadata prefix reported by list mode.
  W="$WORK/det_$name"
  rm -rf "$W" && mkdir -p "$W" && cd "$W" || exit 2
  cp "$corpus" input.dat && $MCM -f8 input.dat a1.mcm >/dev/null 2>&1
  cp "$corpus" input.dat && $MCM -f8 input.dat a2.mcm >/dev/null 2>&1
  rm -f input.dat
  m1=$($MCM l a1.mcm 2>/dev/null | sed -n 's/^Metadata size=\([0-9]*\).*/\1/p')
  m2=$($MCM l a2.mcm 2>/dev/null | sed -n 's/^Metadata size=\([0-9]*\).*/\1/p')
  total=$((total + 1))
  if [ -n "$m1" ] && [ -n "$m2" ] &&
     cmp -s <(tail -c +$((m1 + 1)) a1.mcm) <(tail -c +$((m2 + 1)) a2.mcm); then
    echo "PASS $name determinism (metadata $m1/$m2)"
  else
    echo "FAIL $name determinism"; fails=$((fails + 1))
  fi
done

echo "----"
echo "$((total - fails))/$total passed"
[ "$fails" -eq 0 ]
