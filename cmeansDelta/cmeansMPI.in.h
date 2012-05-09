#ifndef _CMEANS_H_
#define _CMEANS_H_

#include <time.h>

// CPU vs GPU
#define ENABLE_MDL $MDL$
#define CPU_ONLY $CPU_ONLY$

// Which GPU device to use
#define DEVICE $DEVICE$

// number of clusters
#define NUM_CLUSTERS $NUM_CLUSTERS$

// number of dimensions
#define NUM_DIMENSIONS $NUM_DIMENSIONS$

// number of elements
#define NUM_EVENTS $NUM_EVENTS$

// input file delimiter (normally " " or "," or "\t")
#define DELIMITER ","
#define LINE_LABELS 1

// Parameters
#define FUZZINESS $FUZZINESS$
#define THRESHOLD $THRESHOLD$
#define K1 $K1$
#define K2 $K2$
#define K3 $K3$
#define MEMBER_THRESH $MEMBER_THRESHOLD$
#define TABU_ITER $TABU_ITER$
#define TABU_TENURE $TABU_TENURE$
#define VOLUME_TYPE $VOLUME_TYPE$
#define DISTANCE_MEASURE $DISTANCE_MEASURE$
#define MIN_ITERS $MIN_ITERS$
#define MAX_ITERS $MAX_ITERS$

// Naive O(M^2) or optimized O(M) version of membership computation
#define LINEAR 1

// Prints verbose output during the algorithm, enables DEBUG macro
#define ENABLE_DEBUG 0

// Used to enable regular print outs (such as the Rissanen scores, clustering results)
// This should be enabled for general use and disabled for performance evaluations
#define ENABLE_PRINT 1

// Used to enable output of cluster results to .results and .summary files
#define ENABLE_OUTPUT 0

#if ENABLE_DEBUG
#define DEBUG(fmt,args...) printf(fmt, ##args)
#else
#define DEBUG(fmt,args...)
#endif

#if ENABLE_PRINT
#define PRINT(fmt,args...) printf(fmt, ##args)
#else
#define PRINT(fmt,args...)
#endif

// number of Threads and blocks
#define Q_THREADS 192 // number of threads per block building Q
#define NUM_THREADS $NUM_THREADS$  // number of threads per block
#define NUM_THREADS_DISTANCE 512
#define NUM_THREADS_MEMBERSHIP 512
#define NUM_THREADS_UPDATE 512
#define NUM_BLOCKS NUM_CLUSTERS
#define NUM_NUM NUM_THREADS
#define PI (3.1415926)

// Number of cluster memberships computed by each thread in UpdateCenters
#define NUM_CLUSTERS_PER_BLOCK 4

// Amount of loop unrolling for the distance and membership calculations accross dimensions
#define UNROLL_FACTOR 1

// function definitions

void generateInitialClusters(float* clusters, float* events);

float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex);
float MembershipValueDist(const float* clusters, const float* events, int eventIndex, float distance);
float MembershipValueReduced(const float* clusters, const float* events, int clusterIndex, int eventIndex, int);

float* ParseSampleInput(char* filename);

float FindScoreGPU(float* d_matrix, long config);
float* BuildQGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime);
long TabuSearchGPU(float* d_matrix);
void FreeMatrix(float* d_matrix);
int bitCount (int* n);

void ReportResults(float* events, float* clusters, int count, char* inFileName);
void ReportSummary(float* clusters, int count, char* inFileName);


#endif
