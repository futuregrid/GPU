################################################################################
# Build script for MPI/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################

all:cuda

CC	= nvcc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH = -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<

#OBJ		= matrixMul.o matrixMul_kernel.o
OBJ		= matrixMul.o

#matrixMul_kernel.o: matrixMul_kernel.cu
#	nvcc -c matrixMul_kernel.cu -o matrixMul_kernel.o $(INCLUDE)
matrixMul.o: matrixMul.cu
	nvcc -c matrixMul.cu -o matrixMul.o $(INCLUDE) -lcudart

cuda: $(OBJ)
	$(CC) -o cuda_vecadd $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lcutil_x86_64
clean:
	rm *.o cuda_vecadd
