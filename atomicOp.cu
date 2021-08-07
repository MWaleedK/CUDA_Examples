/*#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<device_atomic_functions.h>
#include<device_launch_parameters.h>
#include<memory.h>

#define totalThreads 1000
#define totalBlocks 100000
#define arraySize 10
#define arrLen(arr)((sizeof(arr)>0)?sizeof(arr)/sizeof(arr[0]):-1)
__global__ void atomicAddFn(int *d_a)
{
	int tid=threadIdx.x + blockIdx.x*blockDim.x;

	// not restricting threads here, incrementing with all threads
	tid = tid%arraySize;
	//d_a[tid]++;--- this is an naive approach, use an atomic operation
	atomicAdd(&d_a[tid],1);
}



void main()
{
	int *d_a;
	int * h_a = (int*)malloc(sizeof(int)*arraySize);

	cudaMalloc((void**)&d_a,sizeof(int)*arraySize);
	cudaMemset(d_a, 0, sizeof(int) + arraySize);
	atomicAddFn<<<totalBlocks/totalThreads,totalThreads>>>(d_a);
	memset(h_a,0,sizeof(int)*arraySize);
	cudaMemcpy(h_a,d_a,sizeof(int)* arraySize,cudaMemcpyDeviceToHost);
	for (int i = 0; i < arraySize; i++)
	{
		printf("%d\n", h_a[i]);
	}

	free(h_a);
	cudaFree(d_a);
}*/