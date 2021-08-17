/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>


#define maxThreads 1023
#define maxBlocks 65534
#define imin(a,b)(a<b?a:b)

__global__ void invertColor(unsigned char* d_pic,int height, int width,int channels)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = (x + y*gridDim.x*blockDim.x);

	//the math cheks out my G
	while (tid<height*width)
	{
		d_pic[tid*channels + 0] =255-d_pic[tid*channels + 0];
		if (channels > 1)
		{
			d_pic[tid*channels + 1] =255- d_pic[tid*channels + 1];
			d_pic[tid*channels + 2] = 255-d_pic[tid*channels + 2];
		}
		tid += (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
	}
	printf("%d", (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y));

}


void main()
{
	std::string picName = "Resources/Highway.jpg";
	cv::Mat h_img =cv::imread(picName, cv::IMREAD_COLOR);
	unsigned char* d_img;
	cudaEvent_t start, stop;
	//float elapsedTime;
	int width = h_img.cols;
	int height= h_img.rows;
	int channels = h_img.channels();
	int totalSize = width*height;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	dim3 block(8, 2);
	//dim3 grid((width+block.x-1)/ block.x, (height + block.y - 1) / block.y);
	dim3 grid(5, 3);
	//cudaEventRecord(start, 0);
	cudaMalloc(&d_img,totalSize*channels);
	
	cudaMemcpy(d_img,h_img.ptr(), totalSize*channels, cudaMemcpyHostToDevice);
	invertColor << <grid,block >> >(d_img,height,width,channels);
	cudaMemcpy(h_img.ptr(),d_img,totalSize*channels,cudaMemcpyDeviceToHost);
	
	//cudaRecordEvent(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime,start,stop);
	
	//std::cout << "\nTime taken by GPU: " << elapsedTime << std::endl;

	cv::imshow("img", h_img);
	cv::waitKey();
	cv::destroyAllWindows();
	system("pause");
	cudaFree(d_img);
}*/