#include "hip/hip_runtime.h"
// Compile with hipcc -O3 --amdgpu-target=gfx908 -I${ROCM_PATH}/roctracer/include -L${ROCM_PATH}/roctracer/lib -lroctx64 test.cpp
 
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <hip/hip_profile.h>
//#include <roctx.h>
#include <dh_comms_dev.h> 
 
#define N 10  //2560
#define num_iters 1 //KAL 1000
 
 
template<int n, int m>
__global__ void kernel(double* x) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
    {
//        #pragma unroll
        for (int i = 0; i < n; ++i)
        {
            x[idx] += i * m;
        }
    }
}

inline void hip_assert(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        fprintf(stderr,"HIP error: %s %s %d\n", hipGetErrorString(err), file, line);
        exit(-1);
    }
}
 
#define hipErrorCheck(f) { hip_assert((f), __FILE__, __LINE__); }
#define kernelErrorCheck() { hipErrorCheck(hipPeekAtLastError()); }
 
int main() {
 
    double* x;
    double* x_h;

    std::cout << "QUICKTEST: Runs 1 kernel\n";
 
    size_t sz = N * sizeof(double);
    hipErrorCheck(hipHostMalloc(&x_h, sz));
 
    memset(x_h, 0, sz);
    hipErrorCheck(hipMallocManaged(&x, sz));
    hipErrorCheck(hipMemset(x, 0, sz));
 
    hipStream_t stream;
    hipErrorCheck(hipStreamCreate(&stream));
 
    int blocks = 80;
    int threads = 32;
    int fact = 1;//100;KAL
    
    hipErrorCheck(hipMemcpyAsync(x, x_h, sz, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,1>), dim3(blocks), dim3(threads), 0, 0, x);
    kernelErrorCheck();
    hipFree(x);
    return 0; 
 
}

