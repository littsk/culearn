#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nvToolsExt.h>
#include <cuda_runtime.h>



typedef double my_T;
const int ds = 1024;
const int num_iter = 100;
const int block_dim = 16;

template <typename T>
__global__ void mm(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C, size_t d)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ((idx < d) && (idy < d))
  {
    T temp = 0;
    for (int i = 0; i < d; i++)
      temp += A[idy * d + i] * B[i * d + idx];
    C[idy * d + idx] = temp;
  }
}

int main(int argc, char *argv[])
{

  int use_pinned = 0;
  if (argc > 1)
    use_pinned = atoi(argv[1]);
  if (use_pinned)
    printf("Using pinned memory\n");
  else
    printf("Using pageable memory\n");
  my_T *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;
  int bs = ds * ds * sizeof(my_T);
  cudaMalloc(&d_A, bs);
  cudaMalloc(&d_B, bs);
  cudaMalloc(&d_C, bs);
  if (use_pinned)
  {
    cudaHostAlloc(&h_A, bs, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, bs, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, bs, cudaHostAllocDefault);
  }
  else
  {
    h_A = (my_T *)malloc(bs);
    h_B = (my_T *)malloc(bs);
    h_C = (my_T *)malloc(bs);
  }
  cudaMemset(d_A, 0, bs);
  cudaMemset(d_B, 0, bs);
  memset(h_C, 0, bs);
  dim3 block(block_dim, block_dim);
  dim3 grid((ds + block.x - 1) / block.x, (ds + block.y - 1) / block.y);
  for (int iter = 0; iter < num_iter; iter++)
  {
    mm<<< grid, block >>>(d_A, d_B, d_C, ds);
    if (iter > 1)
      if (h_C[0] != (my_T)((iter - 2) * (iter - 2) * ds))
        printf("validation failure at iteration %d, was %f, should be %f\n", iter, h_C[0], (my_T)((iter - 2) * (iter - 2) * ds));
    for (int i = 0; i < ds * ds; i++)
    {
      h_A[i] = iter;
      h_B[i] = iter;
    }
    nvtxRangePush("cpy");
    cudaMemcpy(h_C, d_C, bs, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, bs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bs, cudaMemcpyHostToDevice);
    nvtxRangePop();
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}