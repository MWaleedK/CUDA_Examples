/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>

#define N 1024*1024
#define FullSize 20*N

__global__ void kernel(unsigned int* a, unsigned int* b, unsigned int* c)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < N)
	{

		c[tid] = a[tid] + b[tid];
	}
}

int main()
{
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if (!prop.deviceOverlap)
		printf("Device overlap not supported by nVidia device, speedup from streams not possible\n");

	unsigned int *h_a, *h_b, *h_c;
	unsigned int *d_a_0, *d_b_0, *d_c_0;
	unsigned int *d_a_1, *d_b_1, *d_c_1;

	//streamcreated and initialized
	cudaStream_t stream_0,stream_1;
	cudaStreamCreate(&stream_0);
	cudaStreamCreate(&stream_1);

	cudaMalloc((void**)&d_a_0, sizeof(unsigned int)*N);
	cudaMalloc((void**)&d_b_0, sizeof(unsigned int)*N);
	cudaMalloc((void**)&d_c_0, sizeof(unsigned int)*N);
	
	cudaMalloc((void**)&d_a_1, sizeof(unsigned int)*N);
	cudaMalloc((void**)&d_b_1, sizeof(unsigned int)*N);
	cudaMalloc((void**)&d_c_1, sizeof(unsigned int)*N);

	//un-paged memory is essential for stream access
	cudaHostAlloc((void**)&h_a, sizeof(unsigned int)*FullSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_b, sizeof(unsigned int)*FullSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_c, sizeof(unsigned int)*FullSize, cudaHostAllocDefault);
	
	unsigned int h_count = 0;
	for (int i = 0; i < FullSize; i++)
	{
		h_a[i] = rand() % 30;
		h_b[i] = rand() % 30;
		h_count += (h_a[i] + h_b[i]);

	}


	float elapsedTime = 0.0f;
	//timing the GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	//sending data in chunks in streams... the need for pinned memor is understandable now.
	for (int i = 0; i < FullSize; i += N*2)
	{
		cudaMemcpyAsync(d_a_0, h_a + i, sizeof(unsigned int)*N, cudaMemcpyHostToDevice, stream_0);//alternate between streams fro max performance
		cudaMemcpyAsync(d_a_1, h_a + i+N, sizeof(unsigned int)*N, cudaMemcpyHostToDevice, stream_1);
		cudaMemcpyAsync(d_b_0, h_b + i, sizeof(unsigned int)*N, cudaMemcpyHostToDevice, stream_0);
		cudaMemcpyAsync(d_b_1, h_b + i+N, sizeof(unsigned int)*N, cudaMemcpyHostToDevice, stream_1);

		kernel << <N / 256, 256, 0, stream_0 >> >(d_a_0, d_b_0, d_c_0);
		kernel << <N / 256, 256, 0, stream_1 >> >(d_a_1, d_b_1, d_c_1);

		cudaMemcpyAsync(h_c + i, d_c_0, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost, stream_0);
		cudaMemcpyAsync(h_c + i+N, d_c_1, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost, stream_1);
	}


	cudaStreamSynchronize(stream_0);
	cudaStreamSynchronize(stream_1);
	cudaStreamDestroy(stream_0);
	cudaStreamDestroy(stream_1);

	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Total Time: %3.1f\n", elapsedTime);
	
	unsigned int h_newCount = 0;
	for (int i = 0; i < FullSize; i++)
	{
		h_newCount += h_c[i];
	}
	(h_newCount == h_count) ? printf("True\n") : printf("False\n");

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	cudaFree(d_a_0);
	cudaFree(d_b_0);
	cudaFree(d_c_0);
	cudaFree(d_a_1);
	cudaFree(d_b_1);
	cudaFree(d_c_1);



	return 0;
}
*/