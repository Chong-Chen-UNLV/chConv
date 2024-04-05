
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
	
	uint16_t inChBias = outChPerBlock*(layer/(inCh/inChPerBlock));	
	uint16_t outChBias = inChPerBlock*(layer%(inch/inChPerBlock));	
	
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
	weigthBias += warpSize*outCh;
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

	}
	
}


static void calBlocksize(const int inCh, const int outCh, 
		const int width,
		const int height,
		int* widthB, 
		int* heightB,
		int* layer){

	int chConst = (inCh/64)*(outCh/64);
	int hParts, wParts = 1;
	hParts = ceil(((float) height)/heightA);
	while(hParts*(wParts + 1) < SM){
		wParts += 1;
	}

	*widthB = wParts;
	*heightB = hParts;

	*layer = (inCh/inChPerBlock)*(outCh/outChPerBlock);
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

	int rowConst = height/heightA;	
	int widthB, heightB, layer;
	calBlocksize(inCh, outCh, width, height, &widthB, &heightB, &layer);	

	uint8_t J_stride = ceil(((float) width)/widthB);

    dim3 blocksize = dim3(widthB, heightB, layer); 
	uint32_t threadSize =512;//try 4x4x32 per each block
	//every kernel call will finish caclulation of all output channels related to 32 input channels. 
	chPool_forward_kernel<<<blocksize, threadSize>>>(input_d, weight_d, output_d, width, height, inCh, outCh);
}

