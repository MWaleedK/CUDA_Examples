/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define threads 1024
#define blocks 1024
#define size blocks*threads


//Do not interchange datatypes this ain't python...

//an example performing reduction operationn
__global__ void kernel(unsigned int* d_in, unsigned int* d_0)
{
	unsigned int myId = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	extern __shared__ unsigned int sData[];
	sData[tid] = d_in[myId];//cop all data to shared memory... faster than global
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sData[tid] += sData[tid+s];
		__syncthreads();
	}
	unsigned int count = 0;
	if (tid==0) {
		d_0[blockIdx.x]=sData[tid];//data back to globbal, this is a partial sum yet to be calculated at
	}
	__syncthreads();
	
		
}


void main()
{
	unsigned int* d_arr;
	unsigned int* d_0;
	
	unsigned int* h_arr = (unsigned int*)malloc(sizeof(unsigned int)*size);
	unsigned int* h_0 = (unsigned int*)malloc(sizeof(unsigned int)*threads);

	unsigned int count = 0;
	for (unsigned int i = 0; i < size; i++)
	{
		h_arr[i] = rand()%246;
		count += h_arr[i];
	}
	//std::cout << count;
	cudaMalloc(&d_arr, size*sizeof(unsigned int));
	cudaMalloc(&d_0, threads*sizeof(unsigned int));
	cudaMemcpy(d_arr,h_arr,sizeof(unsigned int)*size,cudaMemcpyHostToDevice);
	kernel << <blocks, threads,sizeof(unsigned int)*threads >> >(d_arr,d_0);
	cudaMemcpy(h_0,d_0,sizeof(unsigned int)*threads,cudaMemcpyDeviceToHost);
	unsigned int newCount = 0;
	for (unsigned int i = 0; i < threads; i++)
	{
		newCount+= h_0[i];
	}

	if (newCount == count)
		std::cout << "True\n";
	else
		std::cout << "false\n";

	cudaFree(d_arr);
	cudaFree(d_0);
	free(h_arr);
	free(h_0);
	system("pause");
}
*/