#include <stdio.h>


__global__ 
void
broad_cast_from_zero_thread_in_warp(
    int arg)
{
    int laneId = threadIdx.x;
    int value;
    if(laneId == 0){
        value = arg;
    }
    value = __shfl_sync(0xffffffff, value, 0);
    if (value != arg){
        printf("Thread %d failed.\n", threadIdx.x);
    }   
}

int main(
    int argc, 
    char * argv[])
{
    broad_cast_from_zero_thread_in_warp<<<1, 32>>> (1234);
    cudaDeviceSynchronize();
    return 0;
}