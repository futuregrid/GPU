################################################################################
# Build script for OpenMP/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################

all:openmp_cmeans

CC	= g++

INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeansMultiCPU.o MDL.o

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp $(INCLUDE)
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp $(INCLUDE) 
cmeansMultiCPU.o: cmeansMultiCPU.cpp
	$(CC) -c cmeansMultiCPU.cpp -o cmeansMultiCPU.o $(INCLUDE) -fopenmp 

openmp_cmeans: $(OBJ)
	$(CC) -o openmp_cmeans $(OBJ) $(INCLUDE) $(LIBPATH) -fopenmp
clean:
	rm *.o openmp_cmeans
