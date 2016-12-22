#include "cuda_runtime.h"
#include <stdio.h>


int main(void) {
  cudaDeviceProp prop;

  int count;
  cudaGetDeviceCount(&count);

  printf("Device count: %d\n", count);
  
  for(int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);

    printf("Device's name(%d): %s\n", i, prop.name);
    printf("  Total global mem: %zu\n", prop.totalGlobalMem);
    printf("  Shared Mem per Block: %zu\n", prop.sharedMemPerBlock);
    printf("  Register per block: %d\n", prop.regsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Mem Pitch: %zu\n", prop.memPitch);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads dim[0]: %d\n", prop.maxThreadsDim[0]);
    printf("  Max Threads dim[1]: %d\n", prop.maxThreadsDim[1]);
    printf("  Max Threads dim[2]: %d\n", prop.maxThreadsDim[2]);
    printf("  Max Grid Size [0]: %d\n", prop.maxGridSize[0]);
    printf("  Max Grid Size [1]: %d\n", prop.maxGridSize[1]);
    printf("  Max Grid Size [2]: %d\n", prop.maxGridSize[2]);
    printf("  Total Const Mem: %zu\n", prop.totalConstMem);
    printf("  Major: %d\n", prop.major);
    printf("  Minor: %d\n", prop.minor);
    printf("  Texture Alignment: %zu\n", prop.textureAlignment);
    printf("  Device Overlap: %d\n", prop.deviceOverlap);
    printf("  MultiProcessor Count: %d\n", prop.multiProcessorCount);
    printf("  Kernel Exec Timeout Enabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf("  Integrated: %d\n", prop.integrated);
    printf("  Can Map Host Memory: %d\n", prop.canMapHostMemory);
    printf("  Compute Mode: %d\n", prop.computeMode);
    printf("  Max Texture 1D: %d\n", prop.maxTexture1D);
    printf("  Max Texture 2D [0]: %d\n", prop.maxTexture2D[0]);
    printf("  Max Texture 2D [1]: %d\n", prop.maxTexture2D[1]);
    printf("  Max Texture 3D [0]: %d\n", prop.maxTexture3D[0]);
    printf("  Max Texture 3D [1]: %d\n", prop.maxTexture3D[1]);
    printf("  Max Texture 3D [2]: %d\n", prop.maxTexture3D[2]);
    printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
  }

  return 0;
}
