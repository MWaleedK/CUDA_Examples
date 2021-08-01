/*#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void gpuAdd(int h_inp1, int h_inp2, int *d_out)
{
	*d_out = h_inp1 + h_inp2;
}



int main(void)
{

	int h_inp1 = 3;
	int h_inp2 = 8;
	int h_out;
	int *d_out;
	cudaMalloc((void**)&d_out, sizeof(int));
	gpuAdd << <10000, 500 >> >(h_inp1,h_inp2,d_out);
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Your answer is: %d\n", h_out);
	cudaFree(d_out);
	system("Pause");
	return 911;
}*/