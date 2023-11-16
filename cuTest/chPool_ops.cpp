#include <iostream>
#include <string>
#include <iterator>
#include "chPool.h"

static void chPoolCPU(const float* inputTensor,
		const float* weight,
		float* outputTensor,
		const int width,
		const int height,
		const int inCh,
		const int outCh){

	for(int row = 0; row < height; ++row){
		for(int col = 0; col < width; ++col){
			//1x1 weight
			for(int outIt = 0; outIt < outCh; outIt+=32){
				for(int inIt = 0; inIt < inCh; inIt+=32){
					for(int i = 0; i < 32; ++i)
						for(int j = 0; j < 32; ++j)
							outputTensor[row][col][outIt+j]+=inputTensor[row][col][inIt+i]*weight[][];

				}
			}
		}

	}


}


class dummyTensor {

	public: 
		int height, width, ch;
		float* data; //using RAII to avoid memory leak, raw point will not be a problem
		dummyTensor(const int height, const int  width, const int ch, bool rand){


		}

		~dummyTensor(){
			if (height&&width&&ch){
				free(data);
			}

		}

}


		
void torch_launch_chPool(const torch::Tensor &inputTensor_d,
						const torch::Tensor &weightTensor_d,
						torch::Tensor &outputTensor_d,
						const int width,
						const int height,
						const int inCh,
						const int outCh)
                        {
    chPool_forward_C_interface((const float *)inputTensor_d.data_ptr(),
				(const float *)weightTensor_d.data_ptr(),
				(float *)outputTensor_d.data_ptr(),
				const int width,
				const int height,
				const int inCh,
				const int outCh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}
