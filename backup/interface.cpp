#include "chConv.hpp"

//the input/output/weight all need to be square, 
//notice that the program will not work properly if above data is not square
void chConvDepthwiseForward(const int weightSize, 
		const int inChVal,
		const int outChVal,
		const int inWidth,
		const torch::Tensor &weight,
		const torch::Tensor &input,
		torch::Tensor &output)
{
	int blockSize = ;
	int threadSize = ;
	chConvDepthwiseForwardKernel<<<blockSize, threadSize>>>();

}

void chConvDepthwiseBackward(const int weightSize, 
		const int inChVal,
		const int outChVal,
		const int inWidth,
		torch::Tensor &weight,
		const torch::Tensor &input,
		torch::Tensor &output)
{
	//tensor test:
	if(weightSize < 7){
		std::cerr<<"only support larger or equal to 7 conv window"<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	depthwise_kernel_backward_all();

}

void chConv1Forward(const int weightSize, 
		const int batchSize,
		const int inChVal,
		const int outChVal,
		const int inWidth,
		const torch::Tensor &weight,
		const torch::Tensor &input,
		torch::Tensor &output)
{

	if(weightSize != 1){
		std::cerr<<"1x1 conv with error weightsize"<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	int blockSize = ;	
	int threasize = ;

}

void chConv1Backward(const int weightSize, 
		const int batchSize,
		const int inChVal,
		const int outChVal,
		const int inWidth,
		torch::Tensor &weight,
		const torch::Tensor &input,
		torch::Tensor &output)
{

	
	if(weightSize != 1){
		std::cerr<<"1x1 conv with error weightsize"<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	int blockSize = ;	
	int threasize = ;
	chConv1BackwardKernel<<<blockSize, threadSize>>>(batchSize, inChVal, outChVal, weight, input, output);


}



