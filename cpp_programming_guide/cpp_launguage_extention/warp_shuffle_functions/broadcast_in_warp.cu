#include <stdio.h>


__global__ 
void
broad_cast_from_zero_thread_in_warp(
    int arg)
{
    int laneId = threadIdx.x & 0x1f; // landId是线程在一个warp中的id，一个warp有32个线程，所以与上0x1f
    int value = 1;
    if(laneId % 8 == 0){
        value = arg;
    }
    // 注意到，第四个参数是width，相当于把一个warp继续分割，然后每一个subsection的id都是从0开始
    // 如果是8的话，那么就是0,1,2,3,4,5,6,7\0,1,2,3,4,5,6,7\0,1,2,3,4,5,6,7\0,1,2,3,4,5,6,7这样一个区间
    value = __shfl_sync(0xffffffff, value, 0, 8); 
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