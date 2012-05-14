#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include "cmeans.h"

//*H_NUM_EVENTS*H_NUM_DIMENSIONS
__device__ int NUM_EVENTS;
__device__ int NUM_DIMENSIONS;

#include "cmeans_kernel.cu"
#include "MDL.h"
#include "timers.h"
#include <cublas.h>

int H_NUM_EVENTS;
int H_NUM_DIMENSIONS;

/************************************************************************/
/*  Init CUDA                                                           */
/************************************************************************/

#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else

void printCudaError() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }
}

bool InitCUDA(void)
{
    int count = 0;
    int i = 0;
    int device = -1;
    int num_procs = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    printf("There are %d devices.\n",count);
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printf("Device #%d - %s, Version: %d.%d\n",i,prop.name,prop.major,prop.minor);
            // Check if CUDA capable device
            if(prop.major >= 1) {
                if(prop.multiProcessorCount > num_procs) {
                    device = i;
                    num_procs = prop.multiProcessorCount;
                }
            }
        }
    }
    if(device == -1) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }

    device = DEVICE;
    printf("Using Device %d\n",device);
    CUDA_SAFE_CALL(cudaSetDevice(device));

    DEBUG("CUDA initialized.\n");
    return true;
}

#endif

unsigned int timer_io; // Timer for I/O, such as reading FCS file and outputting result files
unsigned int timer_memcpy; // Timer for GPU <---> CPU memory copying
unsigned int timer_cpu; // Timer for processing on CPU
unsigned int timer_gpu; // Timer for kernels on the GPU
unsigned int timer_total; // Total time

/************************************************************************/
/* C-means Main                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{
   
    CUT_SAFE_CALL(cutCreateTimer(&timer_io));
    CUT_SAFE_CALL(cutCreateTimer(&timer_memcpy));
    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_total));
    
    if(!InitCUDA()) {
        return 0;
    }

    CUT_SAFE_CALL(cutStartTimer(timer_total));
    CUT_SAFE_CALL(cutStartTimer(timer_io));
    
    // [program name] [data file]
    if(argc != 2){
        printf("Usage: %s data.csv\n",argv[0]);
        return 1;
    }

    DEBUG("Parsing input file\n");
    float* myEvents = ParseSampleInput(argv[1]);
    cudaMemcpyToSymbol(NUM_EVENTS, &H_NUM_EVENTS, sizeof(int)); 
    cudaMemcpyToSymbol(NUM_DIMENSIONS, &H_NUM_DIMENSIONS, sizeof(int));
 
    if(myEvents == NULL){
        printf("Error reading input file. Exiting.\n");
        return 1;
    }
     
    DEBUG("Finished parsing input file\n");
    
    CUT_SAFE_CALL(cutStopTimer(timer_io));
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
   
    //cublasStatus status;
    //status = cublasInit();
    //if(status != CUBLAS_STATUS_SUCCESS) {
    //    printf("!!! CUBLAS initialization error\n");
    //}

    // Seed random generator, used for choosing initial cluster centers 
    srand((unsigned)(time(0)));
    //srand(42);
    
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*H_NUM_DIMENSIONS);
    float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*H_NUM_DIMENSIONS);
    
    clock_t total_start;
    total_start = clock();

    // Select random cluster centers
    DEBUG("Randomly choosing initial cluster centers.\n");
    generateInitialClusters(myClusters, myEvents);
    
    // Transpose the events matrix
    // Threads within a block access consecutive events, not consecutive dimensions
    // So we need the data aligned this way for coaelsced global reads for event data
    DEBUG("Transposing data matrix.\n");
    float* transposedEvents = (float*)malloc(sizeof(float)*H_NUM_EVENTS*H_NUM_DIMENSIONS);
    for(int i=0; i<H_NUM_EVENTS; i++) {
        for(int j=0; j<H_NUM_DIMENSIONS; j++) {
            transposedEvents[j*H_NUM_EVENTS+i] = myEvents[i*H_NUM_DIMENSIONS+j];
        }
    }
    
    float* memberships = (float*) malloc(sizeof(float)*NUM_CLUSTERS*H_NUM_EVENTS); 
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
    

    int size;
    #if !CPU_ONLY
    DEBUG("Allocating memory on GPU.\n");
    float* d_distanceMatrix;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*H_NUM_EVENTS*NUM_CLUSTERS));
    #if !LINEAR
        float* d_memberships;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_memberships, sizeof(float)*H_NUM_EVENTS*NUM_CLUSTERS));
    #endif

    float* d_center;                                                                              
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_center, sizeof(float)*H_NUM_DIMENSIONS));
    float* d_E;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*H_NUM_EVENTS*H_NUM_DIMENSIONS));
    float* d_C;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*H_NUM_DIMENSIONS));
    float* d_nC;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*H_NUM_DIMENSIONS));    
    float* d_sizes;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sizes, sizeof(float)*NUM_CLUSTERS));    
    float* sizes = (float*) malloc(sizeof(float)*NUM_CLUSTERS);

    size = sizeof(float)*H_NUM_DIMENSIONS*H_NUM_EVENTS;
    CUDA_SAFE_CALL(cudaMemcpy(d_E, transposedEvents, size, cudaMemcpyHostToDevice)); // temporary
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    DEBUG("Copying input data to GPU.\n");
    size = sizeof(float)*H_NUM_DIMENSIONS*H_NUM_EVENTS;
    //CUDA_SAFE_CALL(cudaMemcpy(d_E, myEvents, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_E, transposedEvents, size, cudaMemcpyHostToDevice));
    
    DEBUG("Copying initial cluster centers to GPU.\n");
    size = sizeof(float)*H_NUM_DIMENSIONS*NUM_CLUSTERS;
    CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    #endif
    
    clock_t cpu_start, cpu_stop;
    float diff, max_change;
    cpu_start = clock();
    PRINT("Starting C-means\n");
    float averageTime = 0;
    int iterations = 0;

    // memory size for cluster centers
    size = sizeof(float)*H_NUM_DIMENSIONS*NUM_CLUSTERS;
        
    int num_blocks_distance = H_NUM_EVENTS / NUM_THREADS_DISTANCE;
    if(H_NUM_EVENTS % NUM_THREADS_DISTANCE) {
        num_blocks_distance++;
    }
    int num_blocks_membership = H_NUM_EVENTS / NUM_THREADS_MEMBERSHIP;
    if(H_NUM_EVENTS % NUM_THREADS_DISTANCE) {
        num_blocks_membership++;
    }
    int num_blocks_update = NUM_CLUSTERS / NUM_CLUSTERS_PER_BLOCK;
    if(NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
        num_blocks_update++;
    }

    do{
#if CPU_ONLY
        clock_t start,stop;
        CUT_SAFE_CALL(cutStartTimer(timer_cpu));

        DEBUG("Starting UpdateCenters kernel.\n");
       
        //start = clock();
        //UpdateClusterCentersCPU_Naive(myClusters, myEvents, newClusters);
        //stop = clock();
        //printf("Processing time for Method 1: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
       
        #if !LINEAR 
            start = clock();
            UpdateClusterCentersCPU_Optimized(myClusters, myEvents, newClusters);
            stop = clock();
        DEBUG("Processing time for Quadratic Method: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
        #else 
            start = clock();
            UpdateClusterCentersCPU_Linear(myClusters, myEvents, newClusters);
            stop = clock();
            DEBUG("Processing time for Linear Method: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
        #endif

        DEBUG("Processing time for CPU: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
        averageTime += (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3;
        CUT_SAFE_CALL(cutStopTimer(timer_cpu));
#else
        unsigned int timer = 0;
        CUT_SAFE_CALL(cutCreateTimer(&timer));
        CUT_SAFE_CALL(cutStartTimer(timer));

        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));

        CUT_SAFE_CALL(cutStartTimer(timer_gpu));
        DEBUG("Launching ComputeDistanceMatrix kernel\n");
       ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE >>>(d_center, d_C, d_E, d_distanceMatrix);
        //ComputeDistanceMatrixNoShared<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE >>>(d_C, d_E, d_distanceMatrix);

        #if LINEAR 
            // Optimized, O(M)
            DEBUG("Launching ComputeMembershipLinearMatrix kernel\n");
            ComputeMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP >>>(d_distanceMatrix);
            DEBUG("Launching UpdateClusterCentersGPU kernel\n");
            //UpdateClusterCentersGPU<<< dim3(NUM_CLUSTERS,H_NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix);
            UpdateClusterCentersGPU2<<< dim3(num_blocks_update,H_NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix);
        #else
            // Using unoptimized, O(M^2)
            DEBUG("Launching ComputeMembershipMatrix kernel\n");
            ComputeMembershipMatrix<<< dim3(num_blocks_membership,NUM_CLUSTERS), NUM_THREADS_MEMBERSHIP >>>(d_distanceMatrix, d_memberships);
            DEBUG("Launching UpdateClusterCentersGPU kernel\n");
            //UpdateClusterCentersGPU<<< dim3(NUM_CLUSTERS,H_NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships);
            UpdateClusterCentersGPU2<<< dim3(num_blocks_update,H_NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships);
        #endif
        cudaThreadSynchronize();
       
        // CUBLAS SGEMM: data*transpose(memberships)
        // Transposes are flipped in SGEMM call b/c routine expects column-major (fortran style) data
        /* 
        cublasSgemm('t','n',H_NUM_DIMENSIONS,NUM_CLUSTERS,H_NUM_EVENTS,1.0,d_E,H_NUM_EVENTS,d_distanceMatrix,H_NUM_EVENTS,0.0,d_nC,H_NUM_DIMENSIONS);
        status = cublasGetError();
        if(status != CUBLAS_STATUS_SUCCESS) {  
            printf("Cublas kernel error!\n");
            return 1;
        }
        cudaThreadSynchronize();
        */

        //cublasSgemv('t',H_NUM_EVENTS,H_NUM_DIMENSIONS,1.0,d_E,H_NUM_EVENTS,d_distanceMatrix,1,0,d_nC,1);
 
        DEBUG(cudaGetErrorString(cudaGetLastError()));
        DEBUG("\n");
        CUT_SAFE_CALL(cutStopTimer(timer_gpu));

        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        DEBUG("Copying centers from GPU\n");
        CUDA_SAFE_CALL(cudaMemcpy(newClusters, d_nC, sizeof(float)*NUM_CLUSTERS*H_NUM_DIMENSIONS, cudaMemcpyDeviceToHost));
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));

        // Still need to calculate denominators and divide to get actual centers
        CUT_SAFE_CALL(cutStartTimer(timer_gpu));
        #if LINEAR
            ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_distanceMatrix, d_sizes );
        #else
            ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_memberships, d_sizes );
        #endif
        cudaThreadSynchronize();
        CUT_SAFE_CALL(cutStopTimer(timer_gpu));
        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        cudaMemcpy(sizes,d_sizes,sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
        CUT_SAFE_CALL(cutStartTimer(timer_cpu));
        for(int i=0; i < NUM_CLUSTERS; i++) {
            DEBUG("Size %d: %f\n",i,sizes[i]);
        }

        for(int i=0; i < NUM_CLUSTERS; i++) {
            for(int j=0; j < H_NUM_DIMENSIONS; j++) {
                newClusters[i*H_NUM_DIMENSIONS+j] /= sizes[i];        
            }
        }
        CUT_SAFE_CALL(cutStopTimer(timer_cpu));
        
        CUT_SAFE_CALL(cutStopTimer(timer));
        float thisTime = cutGetTimerValue(timer);
        DEBUG("Iteration time for GPU: %f (ms) \n", thisTime);
        averageTime += thisTime;
        CUT_SAFE_CALL(cutDeleteTimer(timer));

#endif

        CUT_SAFE_CALL(cutStartTimer(timer_cpu));
        
        diff = 0.0;
        max_change = 0.0;
        for(int i=0; i < NUM_CLUSTERS; i++){
            DEBUG("Center %d: ",i);     
            for(int k = 0; k < H_NUM_DIMENSIONS; k++){
                DEBUG("%.2f ",newClusters[i*H_NUM_DIMENSIONS + k]);
                diff += fabs(myClusters[i*H_NUM_DIMENSIONS + k] - newClusters[i*H_NUM_DIMENSIONS + k]);
            max_change = fmaxf(max_change, fabs(myClusters[i*H_NUM_DIMENSIONS + k] - newClusters[i*H_NUM_DIMENSIONS + k]));
                myClusters[i*H_NUM_DIMENSIONS + k] = newClusters[i*H_NUM_DIMENSIONS + k];
            }
            DEBUG("\n");
        }
        DEBUG("Iteration %d, Total Change = %e, Max Change = %e H_NUM_DIMENSIONS:%d\n", iterations, diff, max_change,H_NUM_DIMENSIONS);

        iterations++;
        
        CUT_SAFE_CALL(cutStopTimer(timer_cpu));

    } while((iterations < MIN_ITERS) || (max_change > THRESHOLD && iterations < MAX_ITERS)); 

    #if !CPU_ONLY 
    DEBUG("Computing final memberships\n");
    //CUT_SAFE_CALL(cutStartTimer(timer_gpu));
    ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE >>>(d_center, d_C, d_E, d_distanceMatrix);
    #if LINEAR
        ComputeNormalizedMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP >>>(d_distanceMatrix);
        ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_distanceMatrix, d_sizes );
    #else
        ComputeNormalizedMembershipMatrix<<< dim3(num_blocks_membership,NUM_CLUSTERS), NUM_THREADS_MEMBERSHIP >>>(d_distanceMatrix, d_memberships);
        ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_memberships, d_sizes );
    #endif
    cudaThreadSynchronize();
    //CUT_SAFE_CALL(cutStopTimer(timer_gpu));
    
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    cudaMemcpy(sizes,d_sizes,sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    
    DEBUG("Copying memberships from GPU\n");
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    #if LINEAR
        CUDA_SAFE_CALL(cudaMemcpy(memberships,d_distanceMatrix,sizeof(float)*NUM_CLUSTERS*H_NUM_EVENTS,cudaMemcpyDeviceToHost)); 
    #else
        CUDA_SAFE_CALL(cudaMemcpy(memberships,d_memberships,sizeof(float)*NUM_CLUSTERS*H_NUM_EVENTS,cudaMemcpyDeviceToHost)); 
    #endif
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    #endif

    if(iterations == MAX_ITERS){
        PRINT("Warning: Did not converge to the %f threshold provided\n", THRESHOLD);
        PRINT("Last total change was: %e\n",diff);
        PRINT("Last maximum change was: %e\n",max_change);
    } else {
        PRINT("Converged after iterations: %d\n",iterations); 
    }
    cpu_stop = clock();
    
    CUT_SAFE_CALL(cutStartTimer(timer_io));
    
    averageTime /= iterations;
    printf("\nTotal Processing time: %f (s) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC));
    printf("\n");

    CUT_SAFE_CALL(cutStopTimer(timer_io));
    
    int* finalClusterConfig;
    float mdlTime = 0;

    #if ENABLE_MDL 
        #if CPU_ONLY
            finalClusterConfig = MDL(myEvents, myClusters, &mdlTime, argv[1]);
        #else
            finalClusterConfig = MDLGPU(d_E, d_nC, d_distanceMatrix, &mdlTime, argv[1]);
            mdlTime /= 1000.0; // CUDA timer returns time in milliseconds, normalize to seconds
        #endif
    #else
        finalClusterConfig = (int*) malloc(sizeof(int)*NUM_CLUSTERS);
        memset(finalClusterConfig,1,sizeof(int)*NUM_CLUSTERS);
    #endif

    CUT_SAFE_CALL(cutStartTimer(timer_io));

    // Filters out the final clusters (Based on MDL)
    PRINT("Final Clusters are:\n");
    int newCount = 0;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        if(finalClusterConfig[i]){
            #if !CPU_ONLY
            PRINT("N=%.1f\n",newCount,sizes[i]);
            #endif
            for(int j = 0; j < H_NUM_DIMENSIONS; j++){
                newClusters[newCount * H_NUM_DIMENSIONS + j] = myClusters[i*H_NUM_DIMENSIONS + j];
                PRINT("%.2f\t", myClusters[i*H_NUM_DIMENSIONS + j]);
            }
            newCount++;
            PRINT("\n");
        }
    }
  
    #if ENABLE_OUTPUT 
        ReportSummary(newClusters, newCount, argv[1]);
        ReportResults(myEvents, memberships, newCount, argv[1]);
    #endif
    
    CUT_SAFE_CALL(cutStopTimer(timer_io));
    
    free(newClusters);
    free(myClusters);
    free(myEvents);
    #if !CPU_ONLY
    CUDA_SAFE_CALL(cudaFree(d_E));
    CUDA_SAFE_CALL(cudaFree(d_C));
    CUDA_SAFE_CALL(cudaFree(d_nC));
    #endif

    CUT_SAFE_CALL(cutStopTimer(timer_total));
    printf("\n\n"); 
    printf("Total Time (ms): %f\n",cutGetTimerValue(timer_total));
    printf("I/O Time (ms): %f\n",cutGetTimerValue(timer_io));
    printf("CPU processing Time (ms): %f\n",cutGetTimerValue(timer_cpu));
    printf("GPU processing Time (ms): %f\n",cutGetTimerValue(timer_gpu));
    printf("GPU memcpy Time (ms): %f\n",cutGetTimerValue(timer_memcpy));
    printf("\n\n"); 
    return 0;
}

void generateInitialClusters(float* clusters, float* events){
    int seed;
    srand(time(NULL));
    for(int i = 0; i < NUM_CLUSTERS; i++){
        #if RANDOM_SEED
            seed = rand() % H_NUM_EVENTS;
        #else
            seed = i * H_NUM_EVENTS / NUM_CLUSTERS;
        #endif
        for(int j = 0; j < H_NUM_DIMENSIONS; j++){
            clusters[i*H_NUM_DIMENSIONS + j] = events[seed*H_NUM_DIMENSIONS + j];
        }
    }
    
}

__host__ float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

    float sum = 0;

#if DISTANCE_MEASURE == 0
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum+1e-30);
#endif
#if DISTANCE_MEASURE == 1
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += abs(tmp)+1e-30;
    }
#endif
#if DISTANCE_MEASURE == 2
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = abs(events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp+1e-30;
    }
#endif
    return sum;
}

__host__ float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex){
    //printf("MembershipValue H_NUM_DIMENSIONS:%d\n",H_NUM_DIMENSIONS);
    float myClustDist = CalculateDistanceCPU(clusters, events, clusterIndex, eventIndex);
    float sum =0;
    float otherClustDist;
    for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = CalculateDistanceCPU(clusters, events, j, eventIndex); 
        sum += powf((float)(myClustDist/otherClustDist),(2.0f/(FUZZINESS-1.0f)));
    }
    return 1.0f/sum;
}

void UpdateClusterCentersCPU_Linear(const float* oldClusters, const float* events, float* newClusters){
    //float membershipValue, sum, denominator;
    float membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* memberships = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    for(int i = 0; i < H_NUM_DIMENSIONS*NUM_CLUSTERS; i++) {
        newClusters[i] = 0;
    }
    for(int i = 0; i < NUM_CLUSTERS; i++) {
        numerator[i] = 0;
        denominators[i] = 0;
    }
   
    for(int n = 0; n < H_NUM_EVENTS; n++){
        denominator = 0.0f;
        for(int c = 0; c < NUM_CLUSTERS; c++){
            distances[c] = CalculateDistanceCPU(oldClusters, events, c, n);
            numerator[c] = powf(distances[c],2.0f/(FUZZINESS-1.0f))+1e-30; // prevents divide by zero error if distance is really small and powf makes it underflow
            denominator = denominator + 1.0f/numerator[c];
        }
        
        // Add contribution to numerator and denominator
        for(int c = 0; c < NUM_CLUSTERS; c++){
            membershipValue = 1.0f/powf(numerator[c]*denominator,(float)FUZZINESS);
            for(int d = 0; d < H_NUM_DIMENSIONS; d++){
                newClusters[c*H_NUM_DIMENSIONS+d] += events[n*H_NUM_DIMENSIONS+d]*membershipValue;
            }
            denominators[c] += membershipValue;
        }  
    }

    // Final cluster centers
    for(int c = 0; c < NUM_CLUSTERS; c++){
        for(int d = 0; d < H_NUM_DIMENSIONS; d++){
            newClusters[c*H_NUM_DIMENSIONS + d] = newClusters[c*H_NUM_DIMENSIONS+d]/denominators[c];
        } 
    } 
    
    free(numerator);
    free(denominators);
    free(distances);
    free(memberships);
}
void UpdateClusterCentersCPU_Optimized(const float* oldClusters, const float* events, float* newClusters){
    //float membershipValue, sum, denominator;
    float membershipValue;//, denominator;
    float* numerator = (float*)malloc(sizeof(float)*H_NUM_DIMENSIONS*NUM_CLUSTERS);
    float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* memberships = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    for(int i = 0; i < H_NUM_DIMENSIONS*NUM_CLUSTERS; i++)
        numerator[i] = 0;
    for(int i = 0; i < NUM_CLUSTERS; i++)
        denominators[i] = 0;
   
    float sum;
    for(int n = 0; n < H_NUM_EVENTS; n++){
        // Calculate distance to each cluster center
        for(int c = 0; c < NUM_CLUSTERS; c++){
            distances[c] = CalculateDistanceCPU(oldClusters, events, c, n);
        }

        // Convert distances into memberships
        for(int c = 0; c < NUM_CLUSTERS; c++){
            sum = 0;
            for(int i = 0; i < NUM_CLUSTERS; i++){
                sum += powf((float)(distances[c]/distances[i]),(2.0f/(FUZZINESS-1.0f)));
            }
            memberships[c] = 1.0f/sum;
        }
        
        // Add contribution to numerator and denominator
        for(int c = 0; c < NUM_CLUSTERS; c++){
            membershipValue = memberships[c]*memberships[c];
            for(int d = 0; d < H_NUM_DIMENSIONS; d++){
                numerator[c*H_NUM_DIMENSIONS+d] += events[n*H_NUM_DIMENSIONS+d]*membershipValue;
            }
            denominators[c] += membershipValue;
        }  
    }

    // Final cluster centers
    for(int c = 0; c < NUM_CLUSTERS; c++){
        for(int d = 0; d < H_NUM_DIMENSIONS; d++){
            newClusters[c*H_NUM_DIMENSIONS + d] = numerator[c*H_NUM_DIMENSIONS+d]/denominators[c];
        } 
    } 
    
    free(numerator);
    free(denominators);
    free(distances);
    free(memberships);
}


void UpdateClusterCentersCPU_Naive(const float* oldClusters, const float* events, float* newClusters){
    //float membershipValue, sum, denominator;
    float membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*H_NUM_DIMENSIONS);
    float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    
    for(int i = 0; i < NUM_CLUSTERS; i++){
      denominator = 0.0;
      for(int j = 0; j < H_NUM_DIMENSIONS; j++)
        numerator[j] = 0;
      for(int j = 0; j < H_NUM_EVENTS; j++){
        membershipValue = MembershipValue(oldClusters, events, i, j);
        for(int k = 0; k < H_NUM_DIMENSIONS; k++){
          numerator[k] += events[j*H_NUM_DIMENSIONS + k]*membershipValue*membershipValue;
        }
        
        denominator += membershipValue;
      }  
      for(int j = 0; j < H_NUM_DIMENSIONS; j++){
          newClusters[i*H_NUM_DIMENSIONS + j] = numerator[j]/denominator;
      }  
    }
    
    free(numerator);
    free(denominators);
    free(distances);
}

float* readBIN(char* f) {
    FILE* fin = fopen(f,"rb");
    int nevents,ndims;
    fread(&nevents,4,1,fin);
    fread(&ndims,4,1,fin);
    int num_elements = (ndims)*(nevents);
    printf("Number of rows: %d\n",nevents);
    printf("Number of cols: %d\n",ndims);
    float* data = (float*) malloc(sizeof(float)*num_elements);
    fread(data,sizeof(float),num_elements,fin);
    fclose(fin);
    return data;
}


float* readCSV(char* filename) {
    FILE* myfile = fopen(filename, "r");
    char myline[10000];
    
    if(myfile == NULL){
        printf("Error: File DNE\n");
        return NULL;
    }

    H_NUM_EVENTS = 0;
    while (fgets(myline, 10000, myfile) != NULL)
       H_NUM_EVENTS ++;

    H_NUM_DIMENSIONS = 0;
    if (strtok(myline, DELIMITER) != 0)
        {
            H_NUM_DIMENSIONS++;
            while (strtok(NULL, DELIMITER) != 0)
               H_NUM_DIMENSIONS++;
        }
    //rewind(myfile);
    fclose(myfile);
    //char myline[10000];
    printf("Reading Input:%s  NUM_EVENTS:%d, NUM_DIMMENSIONS:%d\n",filename,H_NUM_EVENTS,H_NUM_DIMENSIONS); 
    float* retVal = (float*)malloc(sizeof(float)*H_NUM_EVENTS*H_NUM_DIMENSIONS);
    myfile = fopen(filename, "r");
    #if LINE_LABELS
        //fgets(myline, 10000, myfile);
        for(int i = 0; i < H_NUM_EVENTS; i++){
            fgets(myline, 10000, myfile);
            retVal[i*H_NUM_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
            for(int j = 1; j < H_NUM_DIMENSIONS; j++){
                retVal[i*H_NUM_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
            }
        }
    #else
        for(int i = 0; i < H_NUM_EVENTS; i++){
            fgets(myline, 10000, myfile);
            retVal[i*H_NUM_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
            for(int j = 1; j < H_NUM_DIMENSIONS; j++){
                retVal[i*H_NUM_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
            }
        }
    #endif
    fclose(myfile);
    return retVal;
}

float* ParseSampleInput(char* f){
    int length = strlen(f);
    printf("File Extension: %s\n",f+length-3);
    if(strcmp(f+length-3,"bin") == 0) {
        return readBIN(f);
    } else {
        return readCSV(f);
    }
}


void FreeMatrix(float* d_matrix){
    CUDA_SAFE_CALL(cudaFree(d_matrix));
}

float* BuildQGPU(float* d_events, float* d_clusters, float* d_distanceMatrix, float* mdlTime){
    float* d_matrix;
    int size = sizeof(float) * NUM_CLUSTERS*NUM_CLUSTERS;

    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    cudaMalloc((void**)&d_matrix, size);
    printCudaError();
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    CUT_SAFE_CALL(cutStartTimer(timer_gpu));

    dim3 grid(NUM_CLUSTERS, NUM_CLUSTERS);
    printf("Launching Q Matrix Kernel\n");
    CalculateQMatrixGPUUpgrade<<<grid, Q_THREADS>>>(d_events, d_clusters, d_matrix, d_distanceMatrix);
    cudaThreadSynchronize();
    printCudaError();

    CUT_SAFE_CALL(cutStopTimer(timer_gpu));
    

    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    float* matrix = (float*)malloc(size);
    printf("Copying results to CPU\n");
    cudaError_t error = cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    printCudaError();
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));

    CUT_SAFE_CALL(cutStopTimer(timer));
    *mdlTime = cutGetTimerValue(timer);
    printf("Processing time for GPU: %f (ms) \n", *mdlTime);
    CUT_SAFE_CALL(cutDeleteTimer(timer));
        
    FreeMatrix(d_matrix);

    printf("Q Matrix:\n");
    for(int row=0; row < NUM_CLUSTERS; row++) {
        for(int col=0; col < NUM_CLUSTERS; col++) {
            printf("%f ",matrix[row*NUM_CLUSTERS+col]);
        }
        printf("\n");
    }
    return matrix;
}

