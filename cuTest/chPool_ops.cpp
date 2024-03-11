#include <iostream>
#include <stdlib.h>
#include <string>
#include <iterator>
#include <time.h>
#include <chrono>
#include "chPool.hpp"

class dummyTensor_d;

class dummyTensor {

	public: 
		int height = 0, width = 0, ch = 0;
		int size;
		float* data; //using RAII to avoid memory leak, raw point will not be a problem
		dummyTensor(const int height_in, const int  width_in, const int ch_in, bool random = true) 
			: height{height_in}, width{width_in}, ch{ch_in}
		{
			size = height*width*ch;
			data = new float[size];
			if(random == true){
				sleep(1);
				srand(time(NULL));
				for(int i = 0; i < size; ++i) data[i] = (float) (rand()%2000-1000)/1000;			
			} else {
				for(int i = 0; i < size; ++i) data[i] = 0;
			}
		}
		//construct a host tensor from a device tensor
		dummyTensor(dummyTensor_d* deviceTensor);
		

		~dummyTensor(){
			if (height&&width&&ch){
				delete[] data;	
			}

		}

};

class dummyTensor_d {

	public: 
		int height, width, ch;
		int size;
		float* data_d;
		dummyTensor_d(dummyTensor* hostTensor){
			height = hostTensor->height;
			width = hostTensor->width;
			ch = hostTensor->ch;
			size = height*width*ch;
			if(hostTensor->data != NULL && size > 0){
				cudaMalloc((void **) &(data_d), size*sizeof(float));
				cudaMemcpy(data_d, hostTensor->data, size*sizeof(float), cudaMemcpyHostToDevice);
			}

		}
		//create a device tensor with no specified initial value
		dummyTensor_d(const int height_in, const int  width_in, const int ch_in)
			: height{height_in}, width{width_in}, ch{ch_in}
		{
			size = height_in*width_in*ch_in;	
			cudaMalloc((void **) &(data_d), size*sizeof(float));
		}

		~dummyTensor_d(){
			if(size > 0){
				if(data_d == NULL)
					std::cout<<"error before free"<<std::endl;
				cudaFree(data_d);
			}

		}

};

dummyTensor::dummyTensor(dummyTensor_d* deviceTensor){
	height = deviceTensor->height;
	width = deviceTensor->width;
	ch = deviceTensor->ch;
	size = height*width*ch;
	data = new float[size];
	if(deviceTensor->data_d != NULL && size > 0){
		cudaMemcpy(data, deviceTensor->data_d, size*sizeof(float), cudaMemcpyDeviceToHost);
	}
}

static void chPoolCPU(dummyTensor& inputTensor,
		dummyTensor& weight,
		dummyTensor& outputTensor){
		const int width = inputTensor.width;
		const int height = inputTensor.height;
		const int inCh = inputTensor.ch;
		const int outCh = outputTensor.ch;
		int weightBias;
	for(int row = 0; row < height; ++row){
		for(int col = 0; col < width; ++col){
			//1x1 weight
			int idxBiasIn = (row*width+col)*inCh;
			int idxBiasOut = (row*width+col)*outCh;
			for(int inIt = 0; inIt < inCh; inIt+=warpSize){
				for(int outIt = 0; outIt < outCh; outIt+=warpSize){
					//every outIt "step-in" not only meaning 1 input channel to 32 out put channel 
					//but 32 input channel to 32 out put channel, so wee need outIt*warpSize
					weightBias = inIt*outCh+outIt*warpSize;
					for(int j = 0; j < warpSize; ++j){
						for(int outI = 0; outI < warpSize; ++outI){
							//for example, j = 30, for outI = 1, updating output 1 with weight start at 30*32, 
							//the input related to this weight should be input[31] in a virtual input array with size 32 
							if(idxBiasOut +outIt + outI >= outputTensor.size || idxBiasIn+inIt+(j+outI)%32 >= inputTensor.size || weightBias+j*32+outI >= weight.size)
								std::cout<<"out of boundary"<<std::endl;
							outputTensor.data[idxBiasOut +outIt + outI]+=inputTensor.data[idxBiasIn+inIt+(j+outI)%32]*weight.data[weightBias+j*32+outI];

						}
					}
				}
			}
		}

	}

}

static void chPoolGPU(dummyTensor_d& inputTensor_d,
		dummyTensor_d& weight_d,
		dummyTensor_d& outputTensor_d){
		const int width = inputTensor_d.width;
		const int height = inputTensor_d.height;
		const int inCh = inputTensor_d.ch;
		const int outCh = outputTensor_d.ch;
	
		for(int i = 0; i < 100; ++i){
			chPool_forward_C_interface(inputTensor_d.data_d, weight_d.data_d, outputTensor_d.data_d, width, height, inCh, outCh);
			cudaDeviceSynchronize();
		}

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		//auto start = std::chrono::high_resolution_clock::now();
		for(int i = 0; i < 10; ++i){
			chPool_forward_C_interface(inputTensor_d.data_d, weight_d.data_d, outputTensor_d.data_d, width, height, inCh, outCh);
		}
		//cudaDeviceSynchronize();

		//auto elapsed = std::chrono::high_resolution_clock::now() - start;
		//long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		//std::cout<<"step time is "<<microseconds<<" us"<<std::endl;

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout<<"step time is "<<milliseconds<<" ms"<<std::endl;



}

static void compare_data(float* yResult, float* y, const float threshold, const int dimension)
{
        double avgdiff = 0;
        double avgampldiff = 0;
        int k = 0;
        for (int i = 0; i < dimension; ++i)
        {
                double d = fabs(y[i] - yResult[i]);
                double ampl = fmin(fabs(y[i]), fabs(yResult[i]));
                if (d > ampl*threshold)
                {
					if(k < 10){
						std::cout<<"large difference at  " << i << "real data  "<<y[i] <<"vs result  " <<  yResult[i] <<std::endl;
						k++;
					}
				}
				avgdiff += d;
                if(ampl > 0) avgampldiff += d/ampl;
        }
        avgdiff ;
        avgampldiff ;
		std::cout<<"diff is  "<<avgdiff<<"ampldiff is "<< avgampldiff<<std::endl;
}



static void tensorCompare(dummyTensor& input1, dummyTensor& input2){
	
	if(input2.height != input1.height | input2.width != input1.width | input2.ch!= input1.ch){
		std::cout<<"meta data difference"<<std::endl;
		return;
	}
	int size = input1.height*input1.width*input1.ch;
	
	compare_data(input1.data, input2.data, 0.005, size);

}	

int main(){

	//int devId=0;
	//cudaSetDevice(devId);
	
	const int height = 64; 
	const int width = 64;
	const int inCh = 2048;
	const int outCh = 2048;

	dummyTensor inputTensor(width, height, inCh, true);
	dummyTensor outputTensor(width, height, outCh, false);
	dummyTensor weightTensor(inCh, outCh, 1, true);

	dummyTensor_d inputTensor_d(&inputTensor);
	dummyTensor_d weightTensor_d(&weightTensor);
	dummyTensor_d outputTensor_d(width, height, outCh);

	chPoolCPU(inputTensor, weightTensor, outputTensor);
	std::cout<<"cpu finished "<<std::endl;
	chPoolGPU(inputTensor_d, weightTensor_d, outputTensor_d);

	dummyTensor outputTensor2(&outputTensor_d);

	tensorCompare(outputTensor2, outputTensor);


	return 0; 
}
		

