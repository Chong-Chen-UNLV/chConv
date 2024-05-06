
//lets assume an forward with aribtary in and out channel numbers
//for example, from input of  64X64X128 to 64X64X256 there will be:
//16 iterations for each pixel. Because one iteration will deal
//with 32 input channel and 64 output channel (1x1 convolution). 
//while 128 (input channel) leads to 4 iteration on input and 
//each iteration will leads to 4 iteration on output. 

#include "chPool.hpp"

#define FULLMSK 0xffffffff

__global__ void setZero(float* inputTensor, const int size){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id < size) inputTensor[id] = 0;
}

__global__ void chPool_forward_kernel_gen(float* inputTensor,
                            const float* weight,
							float* outputTensor,
							const int tensorHeight,
							const int tensorWidth,
							const int inCh,
							const int outCh,
							const int J_stride)
                            
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
	uint16_t layer;
	__shared__ float weightCache[4096];//64 by 64

	uint8_t I_warp;
	uint8_t J_warp;
	uint8_t J_end;
	int tid = threadIdx.x;
	uint16_t warpLane = tid - ((tid>>5)<<5);
	uint16_t warpIdx = tid>>5;
	J_block = blockIdx.x*J_stride;
	I_block = blockIdx.y*heightA;
	layer = blockIdx.z;
	J_end = J_block + J_stride; 
	if(J_end > tensorWidth) J_end = tensorWidth;
	//there are two layer: input layer and output layer
	//if there are 512 input and 512 output, l0=0:0, l8=0:64
	
	uint16_t inChBias = inChPerBlock*(layer/(outCh/outChPerBlock));	
	uint16_t outChBias = outChPerBlock*(layer%(outCh/outChPerBlock));	
	
	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;
	//if(tid == 511)
	//	printf("I_warp is %d, warpLane is %d, warpIdx is %d\n", I_warp, warpLane, warpIdx);	

	// variable "layer" gives the output offset and weight offset

	int pixelOutOffset;

	int pixelInOffset;//inCh and outCh is global constant value but inIt changes according to iterations. 
	int weightBias = inChBias*outCh + outChBias*warpSize;//
	//every 32 input channel related to 32XoutCh step
	//"in this iteration" every 32 output channel step related to 32*32 weight step 
	float val, outVal=0, outVal11=0;
		
	weightCache[tid] = weight[weightBias + tid]; 
	weightCache[tid + 512] = weight[weightBias + tid + 512]; 
	weightCache[tid + 1024] = weight[weightBias + tid + 1024]; 
	weightCache[tid + 1536] = weight[weightBias + tid + 1536]; 
	weightBias += warpSize*outCh;
	weightCache[tid + 2048] = weight[weightBias + tid]; 
	weightCache[tid + 512 + 2048] = weight[weightBias + tid + 512]; 
	weightCache[tid + 1024 + 2048] = weight[weightBias + tid + 1024]; 
	weightCache[tid + 1536 + 2048] = weight[weightBias + tid + 1536]; 
	__syncthreads();
	
	while(I_warp < tensorHeight && J_warp < J_end){
		
		pixelInOffset = (I_warp*tensorWidth + J_warp)*inCh + inChBias;
		pixelOutOffset = (I_warp*tensorWidth + J_warp)*outCh + outChBias;
			
		val = inputTensor[pixelInOffset+warpLane];
		for (int offset = 0; \
				offset < warpSize; offset += 1) {
			//offset<<5 means offset*32
			outVal += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val, warpLane + offset);
			outVal11 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val, warpLane + offset);

		}

		val = inputTensor[pixelInOffset+warpLane+warpSize];
		for (int offset = 0; \
				offset < warpSize; offset += 1) {
			//offset<<5 means offset*32
			outVal += weightCache[warpLane + 2048 + (offset<<5)] * __shfl_sync(FULLMSK, val, warpLane + offset);
			outVal11 += weightCache[warpLane + 2048 + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val, warpLane + offset);

		}

		atomicAdd(&outputTensor[pixelOutOffset+ warpLane], outVal);
		atomicAdd(&outputTensor[pixelOutOffset+ warpLane + 32], outVal11);
		J_warp += widthA;
		outVal = 0;
		outVal11 = 0;

	}
	
}

__global__ void chPool_forward_kernel_spec(float* inputTensor,
                            const float* weight,
							float* outputTensor,
							const int tensorHeight,
							const int tensorWidth,
							const int inCh,
							const int outCh,
							const int J_stride)
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

	uint16_t I_warp;
	uint16_t J_warp;
	uint16_t J_warpIt;
	int tid = threadIdx.x;
	int warpLane = tid - ((tid>>5)<<5);
	int warpIdx = tid>>5;
	J_block = blockIdx.x*J_stride;
	I_block = blockIdx.y*heightA;
	layer = blockIdx.z;
	int16_t	J_end = J_block + J_stride; 
	if(J_end > tensorWidth) J_end = tensorWidth;
	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;
	//if(tid == 511)
	//	printf("I_warp is %d, warpLane is %d, warpIdx is %d\n", I_warp, warpLane, warpIdx);	

	// variable "layer" gives the output offset and weight offset
	int pixelOutOffset = (I_warp*tensorWidth + J_warp)*outCh + layer*outChPerBlock;

	int pixelInOffset = (I_warp*tensorWidth + J_warp)*inCh;//inCh and outCh is global constant value but inIt changes according to iterations. 
	int pixelInOffsetIt, pixelOutOffsetIt;

	const int weightStep = warpSize*outCh;
	int weightBias = layer*warpSize*outChPerBlock;
	//every 32 input channel related to 32XoutCh step
	//"in this iteration" every 32 output channel step related to 32*32 weight step 
	float val, outVal=0, outVal11=0;
	for(int16_t inIt = 0; inIt < inCh; inIt+=warpSize){
		weightCache[tid] = weight[weightBias + tid]; 
		weightCache[tid + 512] = weight[weightBias + tid + 512]; 
		weightCache[tid + 1024] = weight[weightBias + tid + 1024]; 
		weightCache[tid + 1536] = weight[weightBias + tid + 1536]; 
		pixelInOffsetIt = pixelInOffset;
		pixelOutOffsetIt = pixelOutOffset;
		J_warpIt = J_warp;
		__syncthreads();

		for(int it2 = 0; it2 < 2; it2 ++){
			if(J_warpIt < J_end){ val = inputTensor[pixelInOffsetIt+warpLane];
				//0-31 in->0-32 out
				for (int offset = 0; \
						offset < warpSize; offset += 1) {
					//offset<<5 means offset*32
					outVal += weightCache[warpLane + (offset<<5)] * __shfl_sync(FULLMSK, val, warpLane + offset);

					outVal11 += weightCache[warpLane + (offset<<5) + 1024] * __shfl_sync(FULLMSK, val, warpLane + offset);

				}
				outputTensor[pixelOutOffsetIt + warpLane] += outVal;	
				outputTensor[pixelOutOffsetIt + warpLane + 32] += outVal11;	
			}

			outVal11 = 0;
			outVal = 0;
			pixelInOffsetIt += widthA*inCh; 
			pixelOutOffsetIt += widthA*outCh; 
			J_warpIt += widthA;
			__syncthreads();
		}
			__syncthreads();
		pixelInOffset += warpSize;		
		weightBias += weightStep;
	}

}



static bool calBlocksize(const int inCh, const int outCh, 
		const int width,
		const int height,
		int* widthB, 
		int* heightB,
		int* layer){

	int chConst = (inCh/64)*(outCh/64);
	int outChConst = outCh/64;
	int hParts, wParts = 1;

	
	hParts = ceil(((float) height)/heightA);
	bool atomicFlag = true;
	if(outChConst*hParts > ((float) SM*2)/3 ){
		atomicFlag = false;
	}
	
	if(!atomicFlag){
		
		*layer = outCh/outChPerBlock;
		int totalPart = (*layer)*hParts*wParts;
		while(totalPart > ((float) SM*2)/3 && width/wParts > widthA && totalPart%SM < ((float) SM*2)/3 ){
			wParts += 1;
			totalPart = (*layer)*hParts*wParts;
		}
		*widthB = wParts;
		*heightB = hParts;
		return atomicFlag;
		
	} else {

		while(chConst*hParts*wParts < ((float) SM*2)/3 && width/wParts > widthA){
			wParts += 1;
		}
		//if the efficiency is too small:
		int totalPart = chConst*hParts*(wParts + 1);
		//if initial partition is a large number
		//enlarge the partition to make the efficiency
		//larger than 66%
		while(totalPart > ((float) SM*2)/3 && width/wParts > widthA && totalPart%SM < ((float) SM*2)/3 ){
			wParts += 1;
			totalPart = chConst*hParts*(wParts + 1);
		}


		*widthB = wParts;
		*heightB = hParts;

		*layer = (inCh/inChPerBlock)*(outCh/outChPerBlock);
		return atomicFlag;
	}
}

void chPool_forward_C_interface(float* input_d,
		const float* weight_d,
		float* output_d,
		const int width,
		const int height,
		const int inCh,
		const int outCh) {

	//define the number of blocks, closest to the number of SMs. 
	//constant of channel number:

	int widthB, heightB, layer;
	bool atomicFlag = calBlocksize(inCh, outCh, width, height, &widthB, &heightB, &layer);	
	uint8_t J_stride = ceil(((float) width)/widthB);
	dim3 blocksize = dim3(widthB, heightB, layer); 
	uint32_t threadSize =512;//try 4x4x32 per each block
	
	const int zeroThreadSize = 1024;
	int zeroBlockSize = ceil(((float) width*height*outCh)/zeroThreadSize);;
	setZero<<<zeroBlockSize, zeroThreadSize>>>(output_d, width*height*outCh);
	if(atomicFlag){	

		//printf("J_stride is %d, widthB is %d, heightB is %d, layer is %d\n", J_stride, widthB, heightB, layer);
		//every kernel call will finish caclulation of all output channels related to 32 input channels. 
		chPool_forward_kernel_gen<<<blocksize, threadSize>>>(input_d, weight_d, output_d, width, height, inCh, outCh, J_stride);
		
	} else {

		printf("no atomic flag, J_stride is %d, widthB is %d, heightB is %d, layer is %d\n", J_stride, widthB, heightB, layer);
		chPool_forward_kernel_spec<<<blocksize, threadSize>>>(input_d, weight_d, output_d, width, height, inCh, outCh, J_stride);

	}
}

