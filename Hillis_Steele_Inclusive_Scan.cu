#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>
#include<math.h>

__global__ void inclusiveScan(int* input, int * output,int* result,int space,int step, int steps,bool direction)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int tid = x + y*blockDim.x*gridDim.x;


	int ret = 0;
	if (direction) //odd step came
	{
		if (tid < space)
		{
			ret = output[tid];
			input[tid] = ret;
		}
		else
		{
			ret = output[tid]+output[tid - space];
			input[tid] = ret;
		}
	}
	else //even step
	{
		if (tid < space) //copied as they were
		{
			ret = input[tid];
			output[tid] = ret;
		}
		else //copied after index=x+2^step
		{
			ret = input[tid] + input[tid - space];
			output[tid] = ret;
		}
	}

	if (step == steps - 1)
		result[tid] = ret;
}


int main()
{

	int N = 1024*1024;

	int* d_input;
	int* d_output;
	int* d_result;

	int* h_result = new int[N];
	int* h_input = new int[N];

	for (int i = 0; i < N; i++)
	{
		h_input[i] = 1;
	}

	cudaMalloc(&d_input, sizeof(int)*N);
	cudaMalloc(&d_output, sizeof(int)*N);
	cudaMalloc(&d_result, sizeof(int)*N);


	dim3 threads(1024);
	dim3 blocks(N/1024);

	cudaMemcpy(d_input,h_input,sizeof(int)*N,cudaMemcpyHostToDevice);
	
	int steps = static_cast<int>(log2(static_cast<float>(N)));
	int space = 1;
	for (int step = 0; step < steps; step++)
	{
		bool direction = (step % 2 != 0) ?true :false ;
		inclusiveScan << <blocks,threads >> >(d_input,d_output,d_result,space,step,steps,direction);
		space =space*2;//space=space*2;
	}
	//memCpy
	cudaMemcpy(h_result, d_result, sizeof(int)*N,cudaMemcpyDeviceToHost);

	for (int i = 0; i+1 < N; i++)
	{
		h_input[i+1] = h_input[i] + h_input[i + 1];
	}

	int correct = 0;
	int incorrect = 0;
	for (int i = 0; i < N; i++)
		std::cout << h_result[i] << "\t" << h_input[i] << std::endl;
	//(h_input[i]==h_result[i]) ?correct++ :incorrect++ ;
	//std::cout << "Correct: " << correct << "\tincorrect: " << incorrect<<std::endl;


	delete[]h_result;
	delete[]h_input;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_result);


	return 0;
}
