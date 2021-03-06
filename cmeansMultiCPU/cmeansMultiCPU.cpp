#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include "MDL.h"
#include "cmeansMultiCPU.h"

int NUM_EVENTS;

/************************************************************************/
/* C-means Main using multiple CPU cores                                */
/* lihui@indiana.edu   5/20/2012				        */
/************************************************************************/

float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){
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
}//endif

void UpdateClusterCentersCPU_Linear_OpenMP(const float *oldClusters, const float* events, 
	float* newClusters, float* denominators,
	int start, int end){
    float  membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    //float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* memberships = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    for(int i = 0; i < NUM_DIMENSIONS*NUM_CLUSTERS; i++) {
        newClusters[i] = 0;
    }

    for(int i = 0; i < NUM_CLUSTERS; i++) {
        numerator[i] = 0;
        denominators[i] = 0;
    }//for denominator
    for(int n = start; n < end; n++){
        denominator = 0.0f;
        for(int c = 0; c < NUM_CLUSTERS; c++){
            distances[c] = CalculateDistanceCPU(oldClusters, events, c, n);
            numerator[c] = powf(distances[c],2.0f/(FUZZINESS-1.0f))+1e-30;
 // prevents divide by zero error if distance is really small and powf makes it underflow
            denominator = denominator + 1.0f/numerator[c];
        }//for

        //  Add contribution to numerator and denominator
        for(int c = 0; c < NUM_CLUSTERS; c++){
            membershipValue = 1.0f/powf(numerator[c]*denominator,(float)FUZZINESS);
            for(int d = 0; d < NUM_DIMENSIONS; d++){
                newClusters[c*NUM_DIMENSIONS+d] += events[n*NUM_DIMENSIONS+d]*membershipValue;
            }//for
            denominators[c] += membershipValue;
        }//for
    }//for

    free(numerator);
    free(distances);
    free(memberships);

}//end

void UpdateClusterCentersCPU_Linear(const float* oldClusters, const float* events, float* newClusters){
    float  membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* memberships = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    for(int i = 0; i < NUM_DIMENSIONS*NUM_CLUSTERS; i++) {
        newClusters[i] = 0;
    }

    for(int i = 0; i < NUM_CLUSTERS; i++) {
        numerator[i] = 0;
        denominators[i] = 0;
    }
  
    for(int n = 0; n < NUM_EVENTS; n++){
        denominator = 0.0f;
        for(int c = 0; c < NUM_CLUSTERS; c++){
            distances[c] = CalculateDistanceCPU(oldClusters, events, c, n);
            numerator[c] = powf(distances[c],2.0f/(FUZZINESS-1.0f))+1e-20;
 // prevents divide by zero error if distance is really small and powf makes it underflow
            denominator = denominator + 1.0f/numerator[c];
        }

        // Add contribution to numerator and denominator
        for(int c = 0; c < NUM_CLUSTERS; c++){
            membershipValue = 1.0f/powf(numerator[c]*denominator,(float)FUZZINESS);
            for(int d = 0; d < NUM_DIMENSIONS; d++){
                newClusters[c*NUM_DIMENSIONS+d] += events[n*NUM_DIMENSIONS+d]*membershipValue;
            }
            denominators[c] += membershipValue;
        }
    }

    // Final cluster centers
    for(int c = 0; c < NUM_CLUSTERS; c++){
        for(int d = 0; d < NUM_DIMENSIONS; d++){
            newClusters[c*NUM_DIMENSIONS + d] = newClusters[c*NUM_DIMENSIONS+d]/denominators[c];
        }
    }
    free(numerator);
    free(denominators);
    free(distances);
    free(memberships);
}

int main(int argc, char* argv[])
{
    unsigned int timer_io;       // Timer for I/O, such as reading FCS file and outputting result files
    unsigned int timer_total;    // Total time
    unsigned int timer_main_cpu;
    int num_cpus = 4;//omp_get_num_procs();
 
    // [program name]  [data file]
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. %s file.in\n",argv[0]);
        return 1;
    }

    float* myEvents = ParseSampleInput(argv[1]);
    if(myEvents == NULL){
        return 1;
    }
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("---------------------------\n");

    float** tempClusters = (float**) malloc(sizeof(float*)*num_cpus);
    float** tempDenominators = (float**) malloc(sizeof(float*)*num_cpus);
    for(int i=0; i < num_cpus; i++) {
        tempClusters[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
        tempDenominators[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS);
    }
    
    srand((unsigned)(time(0)));
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    //float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);

    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);

    float diff, max_change; // used to track difference in cluster centers between iterations
    float* memberships = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_EVENTS);
    int* finalClusterConfig;
   
    omp_set_num_threads(num_cpus);  // create as many CPU threads as there are CUDA devices

    #pragma omp parallel shared(myClusters,diff,memberships,finalClusterConfig,tempClusters,tempDenominators)
    {
         int tid = omp_get_thread_num();
         int num_cpu_threads = omp_get_num_threads();

         #pragma omp barrier
         printf("CPU thread %d (of %d)\n", tid, num_cpu_threads);
        
         int events_per_cpu = NUM_EVENTS / num_cpus;
         int start = tid*events_per_cpu;
         int finish = (tid+1)*events_per_cpu;
	 if (tid < NUM_EVENTS%num_cpus)
	   finish++;
         int my_num_events = finish-start;

         printf("Starting C-means tid:%d  start:%d  finish:%d\n",tid,start,finish);
         int iterations = 0;

 //	 float* myDenominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
 //      float* myNewClusters = (float*)malloc(sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);

         do{ 
         //cmeans.cpp
         clock_t start,stop;
         DEBUG("Starting UpdateCenters kernel.\n");
    
         #if !LINEAR
            start = clock();
            UpdateClusterCentersCPU_Optimized(myClusters, myEvents, newClusters);
            stop = clock();
         DEBUG("Processing time for Quadratic Method: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
         #else
            start = clock();
            //UpdateClusterCentersCPU_Linear(myClusters, myEvents, newClusters);
	    //void UpdateClusterCentersCPU_Linear_OpenMP(const float *oldClusters, const float* events,
            //float* newClusters, float* denominators,
            //int start, int end)

	 float* myDenominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
         float* myNewClusters = (float*)malloc(sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);

	    UpdateClusterCentersCPU_Linear_OpenMP(myClusters, myEvents, myNewClusters, myDenominators, start,finish);
	    memcpy(tempDenominators[tid],myDenominators,sizeof(float)*NUM_CLUSTERS);
	    memcpy(tempClusters[tid],myNewClusters,sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
            stop = clock();
         DEBUG("Processing time for Linear Method: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
         #endif

         // DEBUG("Processing time for CPU: %f (ms) \n", (float)(stop - start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
	 // averageTime += (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3;
         #pragma omp master
            {
                // Sum up the partial cluster centers (numerators)
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < NUM_DIMENSIONS; d++) {
                            tempClusters[0][c*NUM_DIMENSIONS+d] += tempClusters[i][c*NUM_DIMENSIONS+d];
                        }
                    }
                }

                // Sum up the denominator for each cluster
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        tempDenominators[0][c] += tempDenominators[i][c];
                    }//for
                }//for

		

                // Divide to get the final clusters
                for(int c=0; c < NUM_CLUSTERS; c++) {
		    printf("tempDenominators[%d]:%f\n",c,tempDenominators[0][c]);
                    for(int d=0; d < NUM_DIMENSIONS; d++) {
                        tempClusters[0][c*NUM_DIMENSIONS+d] /= (tempDenominators[0][c]);
                    }
                }
		printf("\n");

                diff = 0.0;
                max_change = 0.0;
                for(int i=0; i < NUM_CLUSTERS; i++){
                    for(int k = 0; k < NUM_DIMENSIONS; k++){
                        diff += fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[0][i*NUM_DIMENSIONS + k]);
      max_change = fmaxf(max_change,fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[0][i*NUM_DIMENSIONS + k]));
                //myClusters[i*NUM_DIMENSIONS + k] = newClusters[i*NUM_DIMENSIONS + k];
                    }
                }//for
                memcpy(myClusters,tempClusters[0],sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);
                //DEBUG("Iteration %d: Total Change = %e, Max Change = %e\n", iterations, diff, max_change);
                //DEBUG("Done with iteration #%d\n", iterations);
            }//#pragma omp master
        #pragma omp barrier

        DEBUG("Iteration %d: Total Change = %e, Max Change = %e\n", iterations, diff, max_change);
        DEBUG("Done with iteration #%d\n", iterations);
        iterations++;
        } while(iterations < MIN_ITERS || (iterations < MAX_ITERS)); 
   
        #if !ENABLE_MDL
            if(tid == 0) {
                finalClusterConfig = (int*) malloc(sizeof(int)*NUM_CLUSTERS);
                memset(finalClusterConfig,1,sizeof(int)*NUM_CLUSTERS);
            }
        #endif
    }//end of omp_parallel block

    int newCount = NUM_CLUSTERS;
    ReportSummary(myClusters, newCount, argv[1]);
 
    #if ENABLE_OUTPUT 
        ReportSummary(newClusters, newCount, argv[1]);
        ReportResults(myEvents, memberships, newCount, argv[1]);
    #endif
    printf("before free myCluster, myEvents\n");    
    //free(newClusters);
    free(myClusters);
    free(myEvents);
    return 0;
}


void generateInitialClusters(float* clusters, float* events){
    int seed;
    srand(time(NULL));
    for(int i = 0; i < NUM_CLUSTERS; i++){
        #if RANDOM_SEED
            seed = rand() % NUM_EVENTS;
        #else
            seed = i * NUM_EVENTS / NUM_CLUSTERS;
        #endif
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            clusters[i*NUM_DIMENSIONS + j] = events[seed*NUM_DIMENSIONS + j];
        }
    }
    
}

float* readBIN(char* f) {
    FILE* fin = fopen(f,"rb");
    int nevents,ndims;
    fread(&nevents,4,1,fin);
    fread(&ndims,4,1,fin);
    int num_elements = NUM_EVENTS*NUM_DIMENSIONS;
    printf("Number of rows: %d\n",nevents);
    printf("Number of cols: %d\n",ndims);
    float* data = (float*) malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
    fread(data,sizeof(float),num_elements,fin);
    fclose(fin);
    return data;
}


float* readCSV(char* filename) {
    printf("reading file:%s\n",filename);
    FILE* myfile = fopen(filename, "r");
    if(myfile == NULL){
        printf("Error: File DNE\n");
        return NULL;
    }
    char myline[1024];

    NUM_EVENTS = 0;
    while (fgets(myline, 10000, myfile) != NULL)
       NUM_EVENTS ++;
    rewind(myfile);

    printf("NUM_EVENTS:%d NUM_DIMENSION:%d\n",NUM_EVENTS,NUM_DIMENSIONS);
    float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);

        for(int i = 0; i < NUM_EVENTS; i++){
            fgets(myline, 1024, myfile);
            retVal[i*NUM_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
            for(int j = 1; j < NUM_DIMENSIONS; j++){
                retVal[i*NUM_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
            }
        }
    printf("finished\n");
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
