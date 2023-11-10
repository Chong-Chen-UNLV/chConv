#include <torch/extension.h>
#include "chPool.h"

		
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
