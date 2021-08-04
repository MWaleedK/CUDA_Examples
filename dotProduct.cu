/*#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_functions.h>

#define imin(a,b)((a<b)?a:b)

const int N =33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+ threadsPerBlock-1) / threadsPerBlock);


__global__ void kernel(float*a,float*b,float*c)
{
	__shared__ float cache[threadsPerBlock];//shared between threads of a block. we have 32 blocks in this example thus, we will have 32 varialbes created for cache each of size 256*float bytes. 
	float temp = 0;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cacheIndex = threadIdx.x;
	while (tid < N)
	{
		temp += a[tid]*b[tid];
		tid += gridDim.x*blockDim.x;
	}

	cache[cacheIndex] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i!=0)
	{
		if(cacheIndex<i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
		//each cache element in all the 32 blocks returning their values at index 0
		printf("blockIdx.x= %d ,threadIdx.x=%d , cache=%f \n", blockIdx.x, threadIdx.x, cache[0]);
	}
}

int main()
{
	
	float* h_a = new float[N];
	float* h_b = new float[N];
	float* h_c = new float[blocksPerGrid];

	float* d_a;
	float* d_b;
	float* d_c;

	cudaMalloc((void**)&d_a, sizeof(float)*N);
	cudaMalloc((void**)&d_b, sizeof(float)*N);
	cudaMalloc((void**)&d_c, sizeof(float)*blocksPerGrid);

	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = 2 * i;
	}

	cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice);

	kernel << <blocksPerGrid, threadsPerBlock >> >(d_a,d_b,d_c);

	cudaMemcpy(h_c, d_c, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
	//Output h_c
	
	//This partial sum calculated on CPU, better than wasting GPU resources
	float c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += h_c[i];
	}

	//CPU sum for verification and compare
	#define sumSquares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g == %.6g\n", c,2*sumSquares((float) (N-1)));


	//Delete all memroAllocs
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete [] h_a;
	delete [] h_b;
	delete[] h_c;
	return 911;
}*/