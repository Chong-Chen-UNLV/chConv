constexpr unsigned int weightCacheSize = 64*32;

//lets assume an 32 to 32 forward
__global__ void conv1d_forward_32(float* inputTensor,
                            const float* weight,
                            float* outputTensor,
                            int inCh,
							int outCh) {

	warpId = ;
	weightCache[tid] = weight[wid]; 
	//first try 32 by 32
    for (int offset = 0; \
            offset < warpSize; offset += 1) {
		outVal += weight[laneId+offset<<4] * __shfl_sync(FULLMSK, val, lanId + offset);
    }
	out[pixelOffset + laneId] = outVal;	
}

//lets assume an forward with aribtary in and out channel numbers
__global__ void conv1d_forward_32(float* inputTensor,
                            const float* weight,
                            float* outputTensor)
	//divide to multiple 32 to 64
	//inCh will always be 32 and outCh will always be 64

	weightBias = (idx*threadSize.x + idy)*iterNum*weightCacheSize;
	weightCache[tid] = weight[weightBias + tid]; 
	weightCache[tid + thtreadSize.x] = weight[weightBias + tid + weightCacheSize/2]; 
	//cache the weight
		
    for (int offset = 0; \
            offset < warpSize; offset += 1) {
		outVal += weight[laneId+offset<<4] * __shfl_sync(FULLMSK, val, lanId + offset);
    }
	out[pixelOffset + laneId] = outVal;	
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}
