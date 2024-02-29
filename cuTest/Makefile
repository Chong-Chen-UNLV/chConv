###############################################################################
#
## Build script for cuda projct
#
#################################################################################

CC=g++
CUDA_HOME=/home/chong/anaconda3/envs/cuda
CU_INC=-I $(CUDA_HOME)/include
CUDA_LIB =-L $(CUDA_HOME)/lib64 -lcudart 

CPPFLAGS= -O3 -lm
CUFLAGS = -O3 -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -std=c++11
CPPFILES = chPool_ops.cpp    
CUFILES = $(wildcard *.cu)
OBJECTS = $(CPPFILES:.cpp=.o)
CU_OBJECTS = $(CUFILES:.cu=.o)

all : $(OBJECTS) $(CU_OBJECTS)
	$(CC) -m64 $^ $(CPPFLAGS) $(CU_INC) $(CUDA_LIB)  -o cuTest.out


$(OBJECTS) : $(CPPFILES)  
	$(CC) -m64 $(CPPFILES) $(CPPFLAGS) $(CU_INC) $(CUDA_LIB) -c

$(CU_OBJECTS) : $(CUFILES)  
	nvcc -c $(CUFILES) $(CUFLAGS) $(CU_INC) $(CUDA_LIB)


clean :
	rm *.o

