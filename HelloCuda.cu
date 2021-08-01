
/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloKernel(void)
{

}

int main()
{
	int count = 0;
	int check=0;
	cudaDeviceProp props;

	cudaGetDeviceCount(&count);
	if(count>0)
	{
		printf("Cuda Devices onboard: %d.\n", count);
		helloKernel << <250, 250 >> >();
		printf("Hello Cuda\n");
		
		for (int i = 0; i < count; i++) 
		{
			cudaGetDeviceProperties(&props, count-1);
			printf("Cuda Device Number: %d\n", count);
			printf("Cuda Device Name : %s\n",props.name);
			printf("CUDA Multiprocessor count: %d\n", props.multiProcessorCount);
			printf("Is %s\n", (check=props.isMultiGpuBoard == 1) ? "MultiGpuBoard\n" : "UniGpuBoard");
			if(check == 1)
			{
				printf("MultiGpuBoardId: %d\n", props.multiGpuBoardGroupID);
			}
			printf("ClockRate: %d MHz\n", props.clockRate/1000);
			printf("MemorClockRate: %d MHz\n", props.memoryClockRate/1000);
			printf("Is %s\n", (props.integrated) ? "Integrated" : "Discrete");
			printf("MaxThreadsPerlock: %d\n\n", props.maxThreadsPerBlock);
		}
	}
	else
	{
		printf("No CUDA capable device found\n");
	}
	system("Pause");
	return 911;
}*/