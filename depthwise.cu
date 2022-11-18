#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void depthwise_kernel_foward(float* inputTensor, 
		float* weightTensor,
		float* outputTensor,
	   	const int8_t width)
{
	extern __shared__ volatile float weightS[];
	__shared__ volatile float inputCache[threadSize];
	float outputCache[4];//8X16 or 4X32 output cache, with 32 warp per block.

	while(){
		inputCache[tid] = ;

		while(tid + threadSize < weightSize){	
			weightS[tid + threadSize] = weightG[tid + threadSize];	
		}

		while(){
			inputCache[tid] = ;
		}	

		for(int8_t i = 0; i < width; ++i){

		}
	}
	

}

__global__ void depthwise_kernel_backward_all(float* outputDelta,
		float* weightTensor,
		float* weightDeltaCache,
		float* inputTensor,
		float* inputDelta,
		const int8_t width)
{
	
	while(){

	}	

}
//backward used sync frequently, may cause performance loss
__global__ void depthwise_kernel_backward_weight(float* outputDelta,
		float* weightDeltaCache,
		float* inputTensor,
		const int8_t width)
{
	float inputCache[4];	
	extern __shared__ volatile float weightS[];
	__shared__ volatile float outputS[];
	while(){
		outputS[] = outputDelta[];
		while(){	
			weightS[] += outputS[]*inputCache[] ;				
			__
		}
		
	}	


}
