################################################################################
# Build script for MPI/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################

all:mpi_cuda

CC	= icc
MPICC	= mpiicc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
INCLUDE   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/ -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeansMPI.o MDL.o

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp $(INCLUDE) -lcudart
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp $(INCLUDE) -lcudart $(INCLUDE)
#cmeansMPI_kernel.o: cmeansMPI_kernel.cu
#	nvcc -c cmeansMPI_kernel.cu -o cmeansMPI_kernel.o $(INCLUDE)
cmeansMPI.o: cmeansMPI.cu
	nvcc -c cmeansMPI.cu -o cmeansMPI.o $(INCLUDE) -lcudart -lmpi -lgomp

mpi_cuda: $(OBJ)
	$(MPICC) -o mpi_cuda_bin $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lmpi -lgomp -lcutil_x86_64
clean:
	rm *.o mpi_cuda_bin
