/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>


float mallocTest(unsigned int size,bool up)
{
	cudaEvent_t start, stop;
	unsigned int *d_var;
	unsigned int *h_var;
	float elapsedTime=0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	h_var = (unsigned int*)malloc(sizeof(unsigned int)*size);
	cudaMalloc((void**)&d_var, size*sizeof(unsigned int));

	cudaEventRecord(start,0);
	for (int i = 0; i < 100; i++)
	{
		if (up)
			cudaMemcpy(d_var,h_var,sizeof(unsigned int)*size,cudaMemcpyHostToDevice);
		else
			cudaMemcpy(h_var, d_var,  sizeof(unsigned int)*size, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_var);
	free(h_var);


	return elapsedTime;
}

float hostAllocTest(unsigned int size, bool up)
{
	cudaEvent_t start, stop;
	unsigned int *d_var;
	unsigned int *h_var;
	float elapsedTime=0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaHostAlloc((void**)&h_var, sizeof(unsigned int)*size, cudaHostAllocDefault);
	cudaMalloc((void**)&d_var, size*sizeof(unsigned int));

	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; i++)
	{
		if (up)
			cudaMemcpy(d_var, h_var, sizeof(unsigned int)*size, cudaMemcpyHostToDevice);
		else
			cudaMemcpy(h_var, d_var, sizeof(unsigned int)*size, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_var);
	cudaFreeHost(h_var);

	return elapsedTime;
}


int main()
{
	unsigned int size = 10 * 1024 * 1024;
	float elapsedTime=0;
	float MB = (float)100 * size*sizeof(unsigned int) / 1024 / 1024;

	elapsedTime = mallocTest(size,true);
	printf("Malloc up_transfer(MB/s): %3.1f\n", MB / (elapsedTime / 1000));
	elapsedTime = hostAllocTest(size, true);
	printf("cudaHostAlloc up_transfer(MB/s): %3.1f\n", MB / (elapsedTime / 1000));
	
	elapsedTime = mallocTest(size, false);
	printf("Malloc down_transfer(MB/s): %3.1f\n", MB / (elapsedTime / 1000));
	elapsedTime = hostAllocTest(size, false);
	printf("cudaHostAlloc down_transfer(MB/s): %3.1f\n", MB/(elapsedTime/1000));
}*/