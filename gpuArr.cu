/*#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


#define N 40000; //ThreadsLaunched

__global__ void gpuArrSet(long *d_in1, long *d_in2)
{
	int M = N;
	long tID = blockIdx.x;
	if(tID<M)
	{
		d_in1[tID] = (-tID);
		d_in2[tID] = (2 * tID);
	}
}

__global__ void gpuArrAdd(long *d_in1, long *d_in2, long*d_out)
{
	int M = N;
	long tID = blockIdx.x;
	if (tID < M)
	{
		d_out[tID] = d_in2[tID] + d_in1[tID];
	}
		
}

int main()
{
	long* d_in1;
	long* d_in2;
	long* d_out;
	int M = N;

	cudaMalloc((void**)&d_in1, sizeof(long)*M);
	cudaMalloc((void**)&d_in2, sizeof(long)*M);
	cudaMalloc((void**)&d_out, sizeof(long)*M);
	long* h_out =(long*) malloc(sizeof(long)*M);
	gpuArrSet << <M , 1>> >(d_in1,d_in2);
	gpuArrAdd << <M , 1>> >(d_in1, d_in2, d_out);

	cudaDeviceSynchronize();
	cudaMemcpy(h_out, d_out, sizeof(int)*M, cudaMemcpyDeviceToHost);
	printf("Output: ");
	for (int i = 0; i < M; i++)
	{
		printf("%ld\n", h_out[i]);
	}
	free(h_out);
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	system("Pause");
	return 911;
}*/