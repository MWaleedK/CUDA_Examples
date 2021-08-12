#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>

#define DIM 1024
#define totalThreads 16
#define totalBlocks DIM/16
#define PI 3.1415

__global__ void kernel(unsigned char* d_img, int channels)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int tid = x + y*gridDim.x*blockDim.x;

	__shared__ float cache[totalThreads][totalThreads];
	const float period=128.0f;
	cache[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI / period) + 1.0f) *(sinf(y*2.0f*PI / period) + 1.0f) / 4.0f;
	__syncthreads();
	if (tid < DIM*DIM)//just incase we launch man threads that tid>totalPixelsOnImg
	{
		d_img[tid * 3 + 0] = 0;
		d_img[tid * 3 + 1] = cache[(totalThreads-1) - threadIdx.x][(totalThreads - 1) - threadIdx.y];
		d_img[tid * 3 + 2] = 0;
	}

}

void main()
{
	unsigned char* d_img;// variable for deice data
	cv::Mat h_img(DIM, DIM, CV_8UC3, cv::Scalar(255, 255, 255));//OpenCV black image, 3 cannel
	cv::Mat img(DIM,DIM,CV_8UC3,cv::Scalar(0,0,255));//BGR, 3 channel image

	int channels = img.channels();//getting channel count
	size_t imgSize= img.rows*img.cols*channels;//total ImageSize
	cudaMalloc((void**)&d_img,imgSize);//allocate memory to GPU
	cudaMemcpy(d_img,img.ptr(), imgSize, cudaMemcpyHostToDevice);//copy data to GPU variable

	dim3 grid(totalBlocks, totalBlocks);//(64,64)
	dim3 threads(totalThreads, totalThreads);//(16,16)

	kernel << <grid, threads >> >(d_img,channels);

	cudaMemcpy(h_img.ptr(),d_img, imgSize, cudaMemcpyDeviceToHost);
	cv::imshow("img", h_img);
	cv::waitKey();
	cv::destroyAllWindows();
	system("pause");
}