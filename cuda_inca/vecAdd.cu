/*
 *  CUDA code for GPU manual pages on FG
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vecadd_kernel.cu"

    //This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    //This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void doVecAdd(int argc, char** argv);
void randomInit(float*, int);

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf(msg);
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{

    int num_gpus = 0;       // number of CUDA GPUs
    cudaGetDeviceCount(&num_gpus);
    if(num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }
    // display CPU and GPU configuration
    printf("number of CUDA devices:\t%d\n", num_gpus);
    printf("cuda inca code start...\n");
    doVecAdd(argc, argv);
    printf("cuda inca code finished\n");
}

////////////////////////////////////////////////////////////////////////////////
// host function
////////////////////////////////////////////////////////////////////////////////

void doVecAdd(int argc, char** argv)
{
    int size = 32;
    int devID;
    cudaDeviceProp props;

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    int block_size = (props.major < 2) ? 16 : 32;
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
	
    uiWA = size;
    uiHA = size;
    uiWB = size;
    uiHB = size;
    uiWC = size;
    uiHC = size;

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    srand(2012);
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A, *d_B, *d_C;
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float* h_C      = (float*) malloc(mem_size_C);
    float* h_CUBLAS = (float*) malloc(mem_size_C);
    checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));
    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(uiWC / threads.x, uiHC / threads.y);

    if (block_size == 16) {
            matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
    } else {
            matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
    }
    cudaDeviceSynchronize();

    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    cudaDeviceReset();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

