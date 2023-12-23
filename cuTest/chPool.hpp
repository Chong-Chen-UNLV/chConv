
#ifndef CHPOOL_H
#define CHPOOL_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

constexpr unsigned int outChPerBlock = 64;
constexpr int warpSize = 32;
constexpr unsigned int weightCacheSize = outChPerBlock*warpSize; //weightCacheSize = 1024
constexpr unsigned int widthA = 8; 
constexpr unsigned int heightA = 8; 
constexpr unsigned int warpPerBlock = widthA*heightA;
constexpr unsigned int threadSize = warpPerBlock*warpSize;

void chPool_forward_C_interface(float* input_d,
		const float* weight_d,
		float* output_d,
		const int width,
		const int height,
		const int inCh,
		const int outCh);



#endif
