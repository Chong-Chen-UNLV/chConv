
#ifndef CHPOOL_H
#define CHPOOL_H

constexpr unsigned int outCh = 64;
constexpr unsigned int warpSize = 32;
constexpr unsigned int weightCacheSize = outCh*warpSize;
constexpr unsigned int widthA = 4; 
constexpr unsigned int heightA = 4; 
constexpr unsigned int warpPerBlock = widthA*heightA;
constexpr unsigned int threadSize = warpPerBlock*warpSize;

void chPool_forward_C_interface(float* input_d,
		const float* weight_d,
		float* output_d,
		const int width,
		const int height,
		const int inCh,
		const int outCh) {

	uint32_t widthB = ceil(((float)width)/widthA);	
	uint32_t heightB = ceil(((float)height)/heightA);	
    dim3 blocksize = dim3(withB, heightB, 1); 
	//check bias

    chPool_forward_kernel<<<grid, block>>>(input_d, weight_d, output_d, width, height, inCh, outCh);
}



#endif
