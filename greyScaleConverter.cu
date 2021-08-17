/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>

#define totalThreads 16  


__global__ void kernel(unsigned char* d_img_in, unsigned char* d_img_out, int channels,int totalSizeImg)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*gridDim.x*blockDim.x;
	
	while (offset<totalSizeImg)
	{
		float grey = 0.0f;
		float blue = d_img_in[offset*channels + 0];
		float green = d_img_in[offset*channels+ 1];
		float red = d_img_in[offset*channels+ 2];
		


		grey = blue*0.11f + green*0.59f + red*0.3f;
		d_img_out[offset + 0] = (grey);
		offset+= (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);//offsetting the threads accessing the grey image
	}
}


int main()
{
	unsigned char* d_img_in;
	unsigned char* d_img_out;


	std::string fileName = "Resources/Highway.jpg";
	cv::Mat h_img_in = cv::imread(fileName, cv::IMREAD_COLOR);



	int channels = h_img_in.channels();
	int imgSizeColor = h_img_in.rows*h_img_in.cols*channels;//total Size of the colored Image
	int totalSize = h_img_in.rows*h_img_in.cols;
	cv::Mat h_img_out(h_img_in.rows, h_img_in.cols, CV_8UC1, cv::Scalar(0));//create the grey image

	int imgSizeGrey = h_img_in.rows*h_img_in.cols;//total Size of the colored Image


	cudaMalloc((void**)&d_img_in, imgSizeColor);//assign memory to GPU variables
	cudaMalloc((void**)&d_img_out, imgSizeGrey);
	dim3 threads(totalThreads, totalThreads);
	dim3 blocks((h_img_in.cols+ totalThreads-1) / totalThreads, (h_img_in.rows+ totalThreads-1) / totalThreads);
	cudaMemcpy(d_img_in, h_img_in.ptr(), imgSizeColor, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_out, h_img_out.ptr(), imgSizeGrey, cudaMemcpyHostToDevice);
	kernel << <blocks, threads >> >(d_img_in, d_img_out, channels,totalSize);//need the dimesnsions of the image to stop thread overwriting 
	cudaMemcpy(h_img_out.ptr(), d_img_out, imgSizeGrey, cudaMemcpyDeviceToHost);


	cv::imshow("img",h_img_out);
	cv::waitKey();
	system("pause");
	cv::destroyAllWindows();


	cudaFree(d_img_in);
	cudaFree(d_img_out);
	return 911;
}*/