#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define N 1024
#define TILE_DIM 16
#define BLOCK_SIZE 256

__global__ void transpose_shared(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x % TILE_DIM;
    int y = blockIdx.y * TILE_DIM + threadIdx.x / TILE_DIM;

    // Read input into shared memory
    if (x < width && y < height) {
        tile[threadIdx.x / TILE_DIM][threadIdx.x % TILE_DIM] = in[y * width + x];
    }

    __syncthreads();

    // Transpose coordinates within tile
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x % TILE_DIM;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.x / TILE_DIM;

    // Write transposed data to output
    if (transposed_x < height && transposed_y < width) {
        out[transposed_y * height + transposed_x] = tile[threadIdx.x % TILE_DIM][threadIdx.x / TILE_DIM];
    }
}

int main() {
    srand(time(0));
    float *h_in = (float*)malloc(N * N * sizeof(float));
    float *h_out = (float*)malloc(N * N * sizeof(float));

    // Fill input with random floats
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_in, *d_out;
    auto err = hipMalloc(&d_in, N * N * sizeof(float));
    err = hipMalloc(&d_out, N * N * sizeof(float));

    err = hipMemcpy(d_in, h_in, N * N * sizeof(float), hipMemcpyHostToDevice);

    // Calculate grid dimensions to cover the entire matrix, even if N is not a multiple of TILE_DIM
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    hipLaunchKernelGGL(transpose_shared, grid, block, 0, 0, d_in, d_out, N, N);

    err = hipMemcpy(h_out, d_out, N * N * sizeof(float), hipMemcpyDeviceToHost);

    err = hipFree(d_in);
    err = hipFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}