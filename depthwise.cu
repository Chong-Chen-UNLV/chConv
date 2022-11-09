#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void depthwise_kernel(float* inputTensor, 
		float* weightTensor,
		float* outputTensor,
	   	const uint8_t width)
{
	extern __shared__ volatile float weightS[];
	__shared__ volatile float inputCache[warpPerBlock*warpSize];
	
	

}
