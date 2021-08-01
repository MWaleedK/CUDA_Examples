
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>

#ifndef DIM
#define DIM 1000
#endif

struct cuComplex {
	float   r;
	float   i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y, float* d_val) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(d_val[0], d_val[1]);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i<200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel(unsigned char *ptr,float* d_val) {
	// map from blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int juliaValue = julia(x, y, d_val);
	ptr[offset * 1 + 0] = 255 * juliaValue;
	
}


//Testing kernel
/*__global__ void kernel(unsigned char* img)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y*gridDim.x;
	if ( (x>100 && x<(gridDim.x-100))  && (y>100 && y<(gridDim.y-100)) )
		img[offset] = 255;

}*/

int main()
{
	//std::string image_path = cv::samples::findFile("starry_night.jpg");
	int inpSiz = 2;
	float* h_val = (float*)malloc(sizeof(float)*inpSiz);
	std::cout << "Input real part:"; std::cin >> h_val[0];
	std::cout << "Input img part:"; std::cin >> h_val[1];

	cv::Mat img(DIM, DIM, CV_8UC1, cv::Scalar(0));
	if (img.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
		return 1;
	}

	//Here goes the code
	const size_t IMGsize = img.step*img.rows;
	unsigned char* d_ptr_in;
	float *d_val;
	cv::Mat h_ptr_out(img.rows, img.cols,CV_8UC1,cv::Scalar(0));
	cudaMalloc((void**)&d_ptr_in, IMGsize);
	cudaMalloc((void**)&d_val, sizeof(float)*2);
	dim3 grid(DIM, DIM);
	
	cudaMemcpy(d_ptr_in, h_ptr_out.ptr(), IMGsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, h_val, sizeof(float)*inpSiz, cudaMemcpyHostToDevice);
	kernel << <grid,1>> >(d_ptr_in,d_val);
	int checkval=cudaMemcpy(h_ptr_out.ptr(), d_ptr_in, IMGsize,cudaMemcpyDeviceToHost);
	if (checkval == 0)
	{
		printf("Ok\n");
	}
	cudaMemcpy(h_val, d_val,sizeof(float)*2,cudaMemcpyDeviceToHost);
	cudaFree(d_ptr_in);
	cudaFree(d_val);
	//
	

	free(h_val);
	imshow("Display window", h_ptr_out);
	cv::waitKey(); // Wait for a keystroke in the window
	return 0;
}

/*#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	int colorWidthStep,
	int grayWidthStep)
{
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	//Launch the color conversion kernel
	bgr_to_gray_kernel << <grid, block >> >(d_input, d_output, input.cols, input.rows, input.step, output.step);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main()
{
	std::string imagePath = "image.jpg";

	//Read input image from the disk
	cv::Mat input = cv::imread(imagePath, cv::IMREAD_COLOR);

	if (input.empty())
	{
		std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	//Call the wrapper function
	convert_to_gray(input, output);

	//Show the input and output
	cv::imshow("Input", input);
	cv::imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}*/