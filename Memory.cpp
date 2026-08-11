/*	MCM file compressor

  Copyright (C) 2013, Google Inc.
  Authors: Mathieu Chartier

  LICENSE

    This file is part of the MCM file compressor.

    MCM is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MCM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MCM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include <cstring>

#include "Memory.hpp"
#include "Util.hpp"

#ifdef WIN32
#include <Windows.h>
#define USE_MALLOC 0
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#define USE_MMAP 1
#define USE_MALLOC 0
#else
#define USE_MALLOC 1
#endif

MemMap::MemMap() : storage(nullptr), size(0) {

}

MemMap::~MemMap() {
  release();
}

void MemMap::resize(size_t bytes) {
  if (bytes == size) {
    zero();
    return;
  }
  release();
  size = bytes;
#if USE_MALLOC
  storage = std::calloc(1, bytes);
#elif defined(USE_MMAP)
  // Anonymous private mappings are zero-filled. The model tables are huge and
  // hit with random accesses, so ask for transparent huge pages to cut dTLB
  // misses (a measured decompression bottleneck); the hint is a no-op where
  // THP is unavailable.
  storage = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (storage == MAP_FAILED) {
    storage = std::calloc(1, bytes);
  } else {
#ifdef MADV_HUGEPAGE
    madvise(storage, size, MADV_HUGEPAGE);
#endif
    mapped_ = true;
  }
#elif WIN32
  storage = (void*)VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
#error UNIMPLEMENTED
#endif
}

void MemMap::release() {
  if (storage != nullptr) {
#if USE_MALLOC
    std::free(storage);
#elif defined(USE_MMAP)
    if (mapped_) {
      munmap(storage, size);
      mapped_ = false;
    } else {
      std::free(storage);
    }
#elif WIN32
    BOOL result = VirtualFree((LPVOID)storage, size, MEM_DECOMMIT);
#else
#error UNIMPLEMENTED
#endif
    storage = nullptr;
  }
  size = 0;
}

void MemMap::zero() {
#if USE_MALLOC
  std::memset(storage, 0, size);
#elif defined(USE_MMAP)
  if (mapped_) {
    // Drop the pages; anonymous mappings refill with zeros on next touch.
    madvise(storage, size, MADV_DONTNEED);
#ifdef MADV_HUGEPAGE
    madvise(storage, size, MADV_HUGEPAGE);
#endif
  } else {
    std::memset(storage, 0, size);
  }
#elif WIN32
  storage = (void*)VirtualAlloc(storage, size, MEM_RESET, PAGE_READWRITE);
#else
  madvise(storage, size, MADV_DONTNEED);
#endif
}
