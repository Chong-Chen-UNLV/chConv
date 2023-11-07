
#ifndef CHPOOL_H
#define CHPOOL_H

constexpr unsigned int outCh = 64;
constexpr unsigned int warpSize = 32;
constexpr unsigned int weightCacheSize = outCh*warpSize;
constexpr unsigned int widthA = 4; 
constexpr unsigned int heightA = 4; 
constexpr unsigned int warpPerBlock = widthA*heightA;
constexpr unsigned int threadSize = warpPerBlock*warpSize;


#endif
