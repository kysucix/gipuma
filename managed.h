#pragma once

#include "helper_cuda.h"

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
      cudaFree(ptr);
  }
};
