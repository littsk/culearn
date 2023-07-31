#include <random>

#define BLOCK_DIM 256

// 一个block负责一行的做法
template<int M, int N>
__global__ void softmax_kernel(float (& input)[M][N], float (& output)[M][N]){
    extern __shared__ float shm[];

    unsigned int row_idx = blockIdx.x;

    float local_max;
    if(threadIdx.x < N){
        local_max = input[row_idx][threadIdx.x];
    }

    // reduce max
    // #pragma unroll
    for(unsigned int i = threadIdx.x; i < N; i += blockDim.x){
        local_max = max(local_max, input[row_idx][i]);
    }
    shm[threadIdx.x] = local_max;
    __syncthreads();
    for(unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(threadIdx.x < offset && threadIdx.x + offset < N){
            shm[threadIdx.x] = max(shm[threadIdx.x], shm[threadIdx.x + offset]);
        }
        // else{
        //     break;
        // }
        __syncthreads();
    }
    
    // row_max = shm[0]
    for(unsigned int i = threadIdx.x; i < N; i += blockDim.x){
        output[row_idx][i] = __expf(input[row_idx][i] - shm[0]);
    }
    __syncthreads();

    // reduce sum
    float local_sum = 0.f;
    for(unsigned int i = threadIdx.x; i < N; i += blockDim.x){
        local_sum += output[row_idx][i];
    }
    shm[threadIdx.x] = local_sum;
    __syncthreads();
    for(unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(threadIdx.x < offset && threadIdx.x + offset < N){
            shm[threadIdx.x] += shm[threadIdx.x + offset];
        }
        // else{
        //     break;
        // }
        __syncthreads();
        
    }

    // row_sum = shm[0]
    for(unsigned int i = threadIdx.x; i < N; i += blockDim.x){
        output[row_idx][i] /= shm[0];
    }
}

template<int M, int N>
void softmax(float (& input)[M][N], float (& output)[M][N]){
    dim3 block(BLOCK_DIM);
    dim3 grid(M);
    softmax_kernel<<<grid, block, BLOCK_DIM * sizeof(float)>>>(input, output);
}

int main(int argc, char * argv[]){
    constexpr int M = 16, N = 512;
    float * input = nullptr, * output = nullptr;
    cudaMallocManaged((void **)(&input), M * N * sizeof(float));
    cudaMallocManaged((void **)(&output), M * N * sizeof(float));

    constexpr int denominator = RAND_MAX >> 4;
    for(int i = 0; i < M * N; ++i){
        input[i] = (float)rand() / denominator;
    }

    float (& real_input)[M][N] = *(float (*)[M][N])input;
    float (& real_output)[M][N] = *(float (*)[M][N])output;

    softmax(real_input, real_output);
    cudaDeviceSynchronize();

    cudaFree(input);
    cudaFree(output);
    
    return 0;
}