################################################################################
# Build script for MPI/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################
# gaussian.cu  gaussian.h  gaussian_kernel.cu  invert_matrix.cpp  invert_matrix.h  Makefile  readData.cpp

all:mpi_cuda_gmm

CC	= icc
MPICC	= mpiicc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
INCLUDE   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/ -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<
OBJ		= gaussian.o invert_matrix.o readData.o

invert_matrix.o: invert_matrix.cpp
	$(CC) -c invert_matrix.cpp $(INCLUDE) -lcudart

readData.o: readData.cpp
	$(CC) -c readData.cpp $(INCLUDE) -lcudart $(INCLUDE)

#cmeansMPI_kernel.o: cmeansMPI_kernel.cu
#	nvcc -c cmeansMPI_kernel.cu -o cmeansMPI_kernel.o $(INCLUDE)

gaussian.o: gaussian.cu
	nvcc -c gaussian.cu -o gaussian.o $(INCLUDE) -lcudart -lmpi -lgomp

mpi_cuda_gmm: $(OBJ)
	$(MPICC) -o mpi_cuda_gmm $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lmpi -lgomp -lcutil_x86_64
clean:
	rm *.o mpi_cuda_gmm
