
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
	uint8_t J_block;
	uint8_t I_block;
	uint8_t layer;
	__shared__ float weightCache[2048];

	uint8_t I_warp;
	uint8_t J_warp;
	int tid = threadIdx.x;
	int warpLane = tid - ((tid>>5)<<5);
	int warpIdx = tid>>5;
	J_block = blockIdx.x*widthA;
	I_block = blockIdx.y*heightA;
	layer = blockIdx.z;

	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;
	//if(tid == 511)
	//	printf("I_warp is %d, warpLane is %d, warpIdx is %d\n", I_warp, warpLane, warpIdx);	

	// variable "layer" gives the output offset and weight offset
	int pixelOutOffset = (I_warp*tensorWidth + J_warp)*outCh + layer*outChPerBlock;
	int pixelOutOffset2 = (tensorWidth/2)*outCh + pixelOutOffset;
	int pixelOutOffset3 = (tensorHeight/2)*tensorWidth*outCh + pixelOutOffset;
	int pixelOutOffset4 = (tensorWidth/2)*outCh + pixelOutOffset3;

	int pixelInOffset = (I_warp*tensorWidth + J_warp)*inCh;//inCh and outCh is global constant value but inIt changes according to iterations. 
	int pixelInOffset2 = pixelInOffset + (tensorWidth/2)*inCh;
	int pixelInOffset3 = pixelInOffset + (tensorHeight/2)*tensorWidth*inCh;
	int pixelInOffset4 = pixelInOffset3 + (tensorWidth/2)*inCh;

	const int weightStep = warpSize*outCh;
	int weightBias = layer*warpSize*outChPerBlock;
	//every 32 input channel related to 32XoutCh step
	//"in this iteration" every 32 output channel step related to 32*32 weight step 
	float val, outVal=0, outVal11=0;
	float val2, outVal2=0, outVal21=0;
	float val3, outVal3=0, outVal31=0;
	float val4, outVal4=0, outVal41=0;
	//float weightR;
	//if(I_warp < tensorHeight && J_warp < tensorWidth){
		for(int inIt = 0; inIt <inCh; inIt+=warpSize){
			weightCache[tid] = weight[weightBias + tid]; 
			weightCache[tid + 512] = weight[weightBias + tid + 512]; 
			weightCache[tid + 1024] = weight[weightBias + tid + 1024]; 
			weightCache[tid + 1536] = weight[weightBias + tid + 1536]; 
			val = inputTensor[pixelInOffset+warpLane];
			val2 = inputTensor[pixelInOffset2+warpLane];
			val3 = inputTensor[pixelInOffset3+warpLane];
			val4 = inputTensor[pixelInOffset4+warpLane];
			__syncthreads();
			//0-31 in->0-32 out
			for (int offset = 0; \
					offset < warpSize; offset += 1) {
				//offset<<5 means offset*32
				outVal += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val, warpLane + offset);
				outVal2 += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val2, warpLane + offset);
				outVal3 += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val3, warpLane + offset);
				outVal4 += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val4, warpLane + offset);

				outVal11 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val, warpLane + offset);
				outVal21 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val2, warpLane + offset);
				outVal31 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val3, warpLane + offset);
				outVal41 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val4, warpLane + offset);
	
			}
			__syncthreads();
			pixelInOffset += warpSize;		
			pixelInOffset2 += warpSize;		
			pixelInOffset3 += warpSize;		
			pixelInOffset4 += warpSize;		
			weightBias += weightStep;
			//0-31 in->32-64 out
			//for (int offset = 0; \
			//		offset < warpSize; offset += 1) {
			//	//offset<<4 means offset*32
			//	outVal += weightCache[1024 + warpLane + offset<<4] * __shfl_sync(FULLMSK, val, warpLane + offset);
			//}

			//outputTensor[pixelOutOffset + warpLane+ warpSize] = outVal;

			//pixelOutOffset += offsetStep;
		}
		outputTensor[pixelOutOffset+ warpLane] = outVal;	
		outputTensor[pixelOutOffset2+ warpLane] = outVal2;	
		outputTensor[pixelOutOffset3+ warpLane] = outVal3;	
		outputTensor[pixelOutOffset4+ warpLane] = outVal4;	
		outputTensor[pixelOutOffset+ warpLane + 32] = outVal11;	
		outputTensor[pixelOutOffset2+ warpLane + 32] = outVal21;	
		outputTensor[pixelOutOffset3+ warpLane + 32] = outVal31;	
		outputTensor[pixelOutOffset4+ warpLane + 32] = outVal41;	
	
	//}
}

void chPool_forward_C_interface(float* input_d,
		const float* weight_d,
		float* output_d,
		const int width,
		const int height,
		const int inCh,
		const int outCh) {

	//if ((width != 40) || (height != 40)){
	//	printf("height is %d, width is %d\n", height, width);
	//	exit(0);
	//}
	uint32_t widthB = ceil(((float) width)/widthA)/2;	
	uint32_t heightB = ceil(((float) height)/heightA)/2;	
	uint32_t layer = outCh/outChPerBlock;//calculate 64 output layer each block 
    dim3 blocksize = dim3(widthB, heightB, layer); 
	uint32_t threadSize =512;//try 4x4x32 per each block
	//every kernel call will finish caclulation of all output channels related to 32 input channels. 
	chPool_forward_kernel<<<blocksize, threadSize>>>(input_d, weight_d, output_d, width, height, inCh, outCh);
}

