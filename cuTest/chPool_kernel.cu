
//lets assume an forward with aribtary in and out channel numbers
//for example, from input of  64X64X128 to 64X64X256 there will be:
//16 iterations for each pixel. Because one iteration will deal
//with 32 input channel and 64 output channel (1x1 convolution). 
//while 128 (input channel) leads to 4 iteration on input and 
//each iteration will leads to 4 iteration on output. 

#include "chPool.hpp"

#define FULLMSK 0xffffffff

__global__ void chPool_forward_kernel(float* inputTensor,
                            const float* weight,
							float* outputTensor,
							const int tensorHeight,
							const int tensorWidth,
							const int inCh,
							const int outCh)
                            
{
	//divide to multiple 32 to 32 
	//we assume the whole area is working like this:
	//each block working with 4X4 pixel area (512 threads)
	//we have (height/4)*(width/4)*(out_channel/32) blocks, 
	//each block dealing with 4x4 area for specified 64 output
	//channel, this method will avoid write conflict between
	//different blocks on the output channels 
	__shared__ int16_t J_block;
	__shared__ int16_t I_block;
	__shared__ int16_t layer;
	__shared__ float weightCache[1024];

	int16_t I_warp;
	int16_t J_warp;
	int16_t tid = threadIdx.x;
	int16_t warpLane = tid - ((tid>>5)<<5);
	int8_t warpIdx = tid>>5;
	if(threadIdx.x == 0){
		J_block = blockIdx.x*widthA;
		I_block = blockIdx.y*heightA;
		layer = blockIdx.z;
	}

	__syncthreads();
	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;
	//if(tid == 511)
	//	printf("I_warp is %d, warpLane is %d, warpIdx is %d\n", I_warp, warpLane, warpIdx);	
	int pixelInOffset = (I_warp*tensorWidth + J_warp)*inCh + tensorHeight*tensorWidth*layer;
	int initOutOffset = (I_warp*tensorHeight + J_warp)*outCh;
	int pixelOutOffset = initOutOffset;
	const int offsetStep = tensorHeight*tensorWidth*warpSize;
	uint16_t weightBias = layer*outCh;
	float val, outVal=0;

	if(I_warp < tensorHeight && J_warp < tensorWidth){
		pixelOutOffset = initOutOffset;
		weightCache[tid] = weight[weightBias + tid]; 
		weightCache[tid + 512] = weight[weightBias + tid + 512]; 
		val = inputTensor[pixelInOffset+warpLane];
		__syncthreads();
		for(uint16_t outIt = 0; outIt < outCh; outIt+=32){
			//0-31 in->0-32 out
			for (int offset = 0; \
					offset < warpSize; offset += 1) {
				//offset<<4 means offset*32
				outVal += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val, warpLane + offset);
			}
			outputTensor[pixelOutOffset+ warpLane] = outVal;	
			pixelOutOffset += offsetStep;
			//0-31 in->32-64 out
			//for (int offset = 0; \
			//		offset < warpSize; offset += 1) {
			//	//offset<<4 means offset*32
			//	outVal += weightCache[1024 + warpLane + offset<<4] * __shfl_sync(FULLMSK, val, warpLane + offset);
			//}

			//outputTensor[pixelOutOffset + warpLane+ warpSize] = outVal;

			//pixelOutOffset += offsetStep;
			weightBias += 32*32;
		}

	}
}

void chPool_forward_C_interface(float* input_d,
		const float* weight_d,
		float* output_d,
		const int width,
		const int height,
		const int inCh,
		const int outCh) {

	uint32_t widthB = ceil(((float)width)/widthA);	
	uint32_t heightB = ceil(((float)height)/heightA);	
	uint32_t layer = outCh/outChPerBlock;//calculate 64 output layer each block 
    dim3 blocksize = dim3(widthB, heightB, layer); 
	uint32_t threadSize =512;//test if it works on compiling
	//every kernel call will finish caclulation of all output channels related to 32 input channels. 
	for(int inIt = 0; inIt < inCh; inIt+=warpSize){
    	chPool_forward_kernel<<<blocksize, threadSize>>>(input_d, weight_d, output_d, width, height, inCh, outCh);
	}
}

