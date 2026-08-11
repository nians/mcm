#!/bin/sh
# Profile-guided build: train on a caller-supplied corpus file (compress +
# decompress, the two hot paths), then rebuild with the profile.
# Usage: ./make_pgo.sh <training-file>
set -e
TRAIN=${1:?usage: ./make_pgo.sh <training-file>}
SRCS="Archive.cpp Huffman.cpp MCM.cpp Memory.cpp Util.cpp Compressor.cpp File.cpp LZ.cpp Tests.cpp"
FLAGS="-DNDEBUG -O3 -fomit-frame-pointer -march=native -std=c++14 -D_FILE_OFFSET_BITS=64"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

g++ $FLAGS -fprofile-generate="$WORK/pgo" -o "$WORK/mcm_train" $SRCS -lpthread
cp "$TRAIN" "$WORK/input.dat"
(cd "$WORK" && ./mcm_train -f8 input.dat train.mcm >/dev/null 2>&1 &&
  rm input.dat && ./mcm_train d train.mcm >/dev/null 2>&1)

g++ $FLAGS -fprofile-use="$WORK/pgo" -fprofile-correction -o mcm $SRCS -lpthread
echo "PGO build done: ./mcm"
