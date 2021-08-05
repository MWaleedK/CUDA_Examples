//FFS, WALEED! PLEASE REMEMBER: blocks DO NOT share memory. Threads in a single block DO.
#include<iostream>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_functions.h>
#include "device_launch_parameters.h"

int const N = 200;

__global__ void staticMemKernel(int *d_a,int n)
{
	__shared__ int cache[N];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tidRev = n - tid - 1;
	
	if (tid < N)
	{
			cache[tid] = d_a[tidRev];
			//printf("(tid: %d,value: %d)\n", tid, cache[tid]);
	}
	__syncthreads();
	
	//printf("(%d,%d)\n", d_a[tid], cache[tidRev]);
	if (tid < N)
	{
		d_a[tid]=cache[tid];
	}


}

__global__ void dynamicMemKernel(int *d_a,int n)
{
	extern __shared__ int cache[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tidRev = n - tid - 1;
	if (tid < N)
	{
		cache[tid] = d_a[tidRev];
		tid += blockDim.x *gridDim.x;
	}
	__syncthreads();

	if (tid < N)
	{
		d_a[tid] = cache[tid];
		
	}
}


void main()
{
	int *h_a = new int[N];
	int *h_b = new int[N];
	int *d_a1;
	int *d_a2;
	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
	}

	cudaMalloc((void**)&d_a1, sizeof(int)*N);
	cudaMalloc((void**)&d_a2,sizeof(int)*N);
	
	cudaMemcpy(d_a1, h_a,N* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a2, h_a,N* sizeof(int), cudaMemcpyHostToDevice);

	staticMemKernel << < 1,N>> >(d_a1,N);

	dynamicMemKernel << <1,N,sizeof(int)*N >> >(d_a2, N);
	cudaMemcpy(h_a,d_a1,sizeof(int)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_a1, sizeof(int)*N, cudaMemcpyDeviceToHost);
	for (int count = 0; count < N; count++)
	{
		if (h_a[count] != h_b[count])
		{
			printf("Error\n");
			break;
		}
		else {
			std::cout <<h_a[count]<<std::endl ;
		}
	}
	


	cudaFree(d_a2);
	cudaFree(d_a1);
	delete[]h_a;
	delete[]h_b;
}