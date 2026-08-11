#!/bin/sh
# Plain optimized build.
g++ -DNDEBUG -O3 -fomit-frame-pointer -march=native -std=c++14 -D_FILE_OFFSET_BITS=64 \
  -o mcm Archive.cpp Huffman.cpp MCM.cpp Memory.cpp Util.cpp Compressor.cpp File.cpp LZ.cpp Tests.cpp -lpthread
