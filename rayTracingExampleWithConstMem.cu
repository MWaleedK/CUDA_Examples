/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>

#define INF 2e10f
#define SPHERES_C 1300
#define threads 16
#define rnd(x) (x * rand() / RAND_MAX)
#define DIM 1024

struct Spheres {
	float x, y, z;
	float radius;
	float r, g, b;
	__device__ float hit(float x_o,float y_o, float *n)
	{
		float dx = x_o - x;
		float dy = y_o - y;
		if (dx*dx + dy*dy < radius*radius)
		{
			float dz = sqrtf(radius*radius-dx*dx-dy*dy);
			*n=dz/sqrtf(radius*radius);
			return dz + z;
		}
		return -INF;
	}
};



__constant__ Spheres s[SPHERES_C];

__global__ void kernel(unsigned char * d_img)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float x_o = (x - DIM / 2);
	float y_o = (y - DIM / 2);

	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i<SPHERES_C; i++) {
		float n;
		float t = s[i].hit(x_o, y_o, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}
	d_img[offset * 3 + 0] = (int)(b * 255);
	d_img[offset * 3 + 1] = (int)(g * 255);
	d_img[offset * 3 + 2] = (int)(r * 255);

}



void main()
{
	cudaEvent_t start, stop;
	float elapsedTime;

	

	
	unsigned char *d_img;
	cv::Mat h_img(DIM,DIM,CV_8UC3,cv::Scalar(0,0,0));

	//save data to a buffer on the Host mem
	Spheres* temp_s = (Spheres*)malloc(sizeof(Spheres)*SPHERES_C);
	for (int i = 0; i < SPHERES_C; i++)
	{
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//img data set on Device
	cudaMalloc(&d_img,(DIM*DIM*h_img.channels()));
	cudaMemcpy(d_img,h_img.ptr() ,(DIM*DIM*h_img.channels()),cudaMemcpyHostToDevice);

	//Spheres data set on Device
	cudaMemcpyToSymbol(s,temp_s, sizeof(Spheres)*SPHERES_C);//send Sphere data to device
	free(temp_s);//free memory

	dim3 grid((DIM) / threads, (DIM) / threads);
	dim3 block(threads, threads);
	//kernel Launch
	kernel << < grid,block>> >(d_img);
	

	cudaMemcpy(h_img.ptr(), d_img, (DIM*DIM*h_img.channels()), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	std::cout <<"\nElapsed Time: " <<elapsedTime << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cv::imshow("img", h_img);
	cv::waitKey();
	cv::destroyAllWindows();
	cudaFree(d_img);
	cudaFree(s);
}
*/