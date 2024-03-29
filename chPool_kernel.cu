
//lets assume an forward with aribtary in and out channel numbers
//for example, from input of  64X64X128 to 64X64X256 there will be:
//16 iterations for each pixel. Because one iteration will deal
//with 32 input channel and 64 output channel (1x1 convolution). 
//while 128 (input channel) leads to 4 iteration on input and 
//each iteration will leads to 4 iteration on output. 
__global__ void chPool_forward_kernel(float* inputTensor,
                            const float* weight,
							const int tensorHeight,
							const int tensorWidth,
							const int inCh
							const int outCh,
                            float* outputTensor)
{
	//divide to multiple 32 to 64
	//we assume the whole area is working like this:
	//each block working with 4X4 pixel area (512 threads)
	//we have (height/4)*(width/4)*(out_channel/64) blocks, 
	//each block dealing with 4x4 area for specified 64 output
	//channel, this method will avoid write conflict between
	//different blocks on the output channels 
	__shared__ int16_t J_block;
	__shared__ int16_t I_block;

	int16_t I_warp;
	int16_t J_warp;
	int16_t warpLane = threadIdx.x - ((threadIdx.x>>5)<<5);
	int8_t warpIdx = threadIdx.x>>5;
	if(threadIdx.x == 0){
		J_block = blockIdx.x*widthA;
		I_block = blockIdx.y*heightA;
	}
	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;

	int pixelInOffset = (I_warp*imageWidth + J_warp)*inCh;
	int initOutOffset = (I_warp*imageWidth + J_warp)*outCh;
	int pixelOutOffset = initOutOffset;
	const int offsetStep = imageHeight*imageWidth*warpSize;
	int waightBias = 0;

	if(I_warp < imageHeight && J_warp < imageWidth){
		for(uint16_t inIt = 0; inIt < inCh; inIt+=32){
			pixelOutOffset = initOffset;
			for(uint16_t outIt = 0; outIt < outCh; outIt+=64){

				weightCache[tid] = weight[weightBias + tid]; 
				weightCache[tid + threadSize.x] = weight[weightBias + tid + weightCacheSize/2]; 

				//0-31 in->0-32 out
				for (int offset = 0; \
						offset < warpSize; offset += 1) {
					//offset<<4 means offset*32
					outVal += weightCache[laneId + offset<<4] * __shfl_sync(FULLMSK, val, lanId + offset);
				}
				out[pixelOutOffset+ laneId] = outVal;	
				pixelOutOffset += offsetStep;
				//0-31 in->32-64 out
				for (int offset = 0; \
						offset < warpSize; offset += 1) {
					//offset<<4 means offset*32
					outVal += weightCache[threadSize.x + laneId + offset<<4] * __shfl_sync(FULLMSK, val, lanId + offset);
				}

				out[pixelOutOffset + laneId + warpSize] = outVal;

				weightBias += 32*64;
				pixelOutOffset += offsetStep;
			}

			pixelInOffset += offsetStep;
		}
	}
}



