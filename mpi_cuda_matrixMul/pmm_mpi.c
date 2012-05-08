/*-------------------------------------------------------------------------------

----------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"
#include "mkl_cblas.h"

char file_name[128];
FILE *fptr;

typedef struct {
   double *local_A;
   double *local_B;
   double *local_C;
   int m;
   int n_threads;
   int tid;
   int rank;
} ThreadsInfo;

typedef struct {
    MPI_Comm  comm;      /* Communicator for entire grid */
    MPI_Comm  row_comm;  /* Communicator for my row      */
    MPI_Comm  col_comm;  /* Communicator for my col      */
    int p;               /* Total number of processes    */
    int s;               /* Order of grid                */
    int my_row;          /* My row number                */
    int my_col;          /* My column number             */
    int my_rank;         /* My rank in the grid comm     */
    int my_world_rank;   /* My rank in the world comm    */
} PROCESS_INFO_T;

/*  Function headers, actual implementation later in file */

double* Initialize_Matrix(PROCESS_INFO_T grid, int n);

void  Print_matrix(char* header, double* local_A, 
                     PROCESS_INFO_T grid, int n, int m);

void Set_to_zero(double* local_A, int m);
void *doCalculate(void *ptr);
void *doTilesCalculate(void *ptr);
void *doDGEMMCalculate(void *ptr);
void *doCUDA(void *ptr);

void Local_matmat(double* local_A,
             double* local_B, double* local_C, int m);

void Local_matmat_pthread( double *local_A, double *local_B, double *local_C, int m,int n_threads,int option, int rank);

/*----------------------------------------------------------
 * The BLAS routine for the local matrix*matrix multiply. 
 * Note the underscore convention; this calls the standard
 * BLAS interface, not the nonstandard non-portable
 * C interface.
 *----------------------------------------------------------
*/

/*===================================*/
int main(int argc, char* argv[]) {
/*===================================*/
    int   p;
    int   my_rank;
    PROCESS_INFO_T  grid;

    double*  local_A;
    double*  local_B;
    double*  local_C;
    double*  temp_mat;
    
    int n;
    int m;
    int option;
    int n_threads;

    int got_local_memory = 1;
    int got_all_memory = 0;
    double t1, t2, Mflops;

    /* Do all of the subcommunicator creation */
    void Setup_grid(PROCESS_INFO_T* grid);

    /* Actually performs the broadcast-multiply-roll algorithm */
    void BMR(int n, PROCESS_INFO_T grid, double* local_A,
             double* local_B, double* local_C,
             double* temp_A,int n_threads,int option);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    sprintf(file_name,"%s","fox_mkl_alamo.log");
    Setup_grid(&grid);
	
    if (argc!=4){
	printf("mpi matrix multiplication parameters:\n");
	printf("[matrix_size] [n_threads] [option]\n");
        printf("option 1) general matrix multiplication\n");
	printf("       2) tiles matrix multiplication (C version)\n");
	printf("       3) cblas matrix multiplication (C BLAS version)\n");
	printf("       4) cublas matrix multiplication (cuda BLAS version)\n");
	exit(-1);	
    }//if

    if (my_rank == 0) {
	(void) sscanf(argv[1], "%d", &n);
        (void) sscanf(argv[2], "%d", &n_threads);
	(void) sscanf(argv[3], "%d", &option);
        //printf("n_threads %d  argc:%d ",n_threads,argc);
    }//if

    /* First make sure that n/s is an integer */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&option, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_threads,1,MPI_INT,0, MPI_COMM_WORLD);

    if (my_rank == 1)
    printf("process[1] n:%d, option:%d, threads:%d\n",n,option,n_threads);

    m = n/grid.s;
    if (m*grid.s != n) {
      if (grid.my_world_rank == 0) {
          printf("\nn must be multiple of s\n");
          printf("Instead you gave n = %d and s = %d. Try \n", n, grid.s);
          printf("it again, but *first* read the documentation, dipshit.\n");
      };
      MPI_Finalize();
      exit(-1);
    }

    /*-----------------*/
    /* Allocate memory */
    /*-----------------*/

    local_A  = Initialize_Matrix(grid, m);
    local_B  = Initialize_Matrix(grid, m);
    local_C  = (double *) malloc(m*m*sizeof(double));
    temp_mat = (double *) malloc(m*m*sizeof(double));

    if ( local_A == NULL || local_B == NULL ||
         local_C == NULL || temp_mat == NULL) got_local_memory = 0;

    /*-------------------------------------------------------------------
     * This trick is actually an idiom in MPI error checking 
     * It forms the product, so if even on process encountered an error,
     * everyone will know it. Knowing who screwed up is irrelevant.
     *--------------------------------------------------------------------*/
    MPI_Allreduce(&got_local_memory, &got_all_memory, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    if (got_all_memory == 0) {
      if (grid.my_world_rank == 0) 
          printf("Memory allocation failed\n");
      MPI_Finalize();
      exit(-1);
    }

    /*-------------------------------*/
    /* Call BMR algorithm, timing it */
    /*-------------------------------*/

    t1 = MPI_Wtime();
    BMR(n, grid, local_A, local_B, local_C, temp_mat,n_threads,option);
    t2 = MPI_Wtime() - t1;
    Mflops = (2.0e-6)*((double)n)*((double)n)*((double)n)/t2;

    if (my_rank == 0) {
        printf("Total Mflop rate: %e \t", Mflops);
        printf("Total job time: %f\n\n", t2);
  	fptr = fopen (file_name,"a");
  	if (fptr!=NULL)
  	{
    	fprintf (fptr,"Size:%d Threads:%d option:%d Total Mflop rate:%e Time:%f\n",n,n_threads,option,Mflops,t2);
    	fclose (fptr);
  	}
    }//if
    MPI_Finalize();
    return 0;
}


/*=========================================*/
void Setup_grid( PROCESS_INFO_T*  grid ) {
/*=========================================*/
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /*-----------------------------*/
    /* Get world communicator rank */
    /*-----------------------------*/
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);
    grid->my_world_rank = old_rank;

    /*---------------------------*/
    /* Number of world processes */
    /*---------------------------*/
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));

    /*---------------------------------------*/
    /* Number of processes in each dimension */
    /*---------------------------------------*/
    grid->s = (int) sqrt((double) grid->p);
    //
    //grid->s scale; of matrix;
    //
    dimensions[0] = dimensions[1] = grid->s;
    if ((grid->s)*(grid->s) != grid->p) {
      if(old_rank == 0) {
        printf("Foulup on grid dimensions; p is not perfect square. \n");
        printf("Don't you know what a perfect square is? \n");
        printf("Don't you wish now you had not slept through math clases \n");
        printf("in fourth grade? Assuming you ever made it \n");
        printf("that far, that is.\n");
      }
      MPI_Finalize();
      exit(-1);
    }

    /*------------------------------------------------------*/
    /* Use circular shift for columns, don't care for rows */
    /*------------------------------------------------------*/
    wrap_around[0] = wrap_around[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /*---------------------------------*/
    /* Split out the row communicators */
    /*---------------------------------*/
    free_coords[0] = 0; 
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->row_comm));

    /*------------------------------------*/
    /* Split out the column communicators */
    /*------------------------------------*/
    free_coords[0] = 1; 
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->col_comm));
}

//local Matrix Multplication for row and column block. 
/*===============================================*/
void BMR( int n, PROCESS_INFO_T     grid, 
        double*  local_A, double*  local_B,
        double*  local_C, double*  temp_A, int n_threads, int option) {
/*===============================================*/

    int k;
    int bcast_root;
    int m;
    int source;
    int dest;
    MPI_Status status;
    //FILE *fptr;
    //char file_name[128];
    double t1,t2,t3,t4;
    double network_a,network_b;
    double cpu_time;

    //sprintf(file_name,"%s","network_cost.log");
    fptr = fopen(file_name,"a");

    m = n/grid.s;
    if (m*grid.s != n) {
      if (grid.my_world_rank == 0) 
         printf("\nn must be multiple of s\n");
         printf("Instead you gave n = %d and s = %d. Try \n", n, grid.s);
         printf("it again, but this time read the comments.\n");
         printf("Or find someone literate, and have that person read.\n");
         printf("them out loud for you.\n");
      MPI_Finalize();
      exit(-1);
    }//if

    /*-----------------------------------------*/
    /* Comment this out for a true C = C + A*B */
    /* instead of C = A*B.                     */
    /*-----------------------------------------*/
    Set_to_zero(local_C, m);

    /*------------------------------------------------------*/  
    /* Here is the only part you have to really think about */  
    /*------------------------------------------------------*/  
    source = (grid.my_row + 1) % grid.s;
    dest = (grid.my_row + grid.s - 1) % grid.s;
    network_a = 0.0;
    network_b = 0.0;

    for (k = 0; k < grid.s; k++) {

	if (grid.my_world_rank==0){
		printf("iteration :%d\n",k);
		t1 = MPI_Wtime();
	}//if

        bcast_root = (grid.my_row + k) % grid.s;

        if (bcast_root == grid.my_col) {
           MPI_Bcast(local_A, m*m, MPI_DOUBLE, bcast_root, grid.row_comm);

	   if (grid.my_world_rank == 0){
	   t2 = MPI_Wtime();
	   network_a += t2-t1;	   
	   printf(" rank[0] broadcast A take:%f iteration:%d\n", t2-t1,k);
	   }//if

	   Local_matmat_pthread(local_A,local_B,local_C, m, n_threads, option, grid.my_world_rank);
	   if (grid.my_world_rank == 0){
 	   t3 = MPI_Wtime();
	   cpu_time += t3-t2;		
	   }//if

        } else {
           MPI_Bcast(temp_A, m*m, MPI_DOUBLE, bcast_root, grid.row_comm);

	   if (grid.my_world_rank == 0){
           t2 = MPI_Wtime();
           network_a += t2-t1;
 	   printf(" rank[0] broadcast A take:%f iteration:%d\n", t2-t1,k);
           }//if

           Local_matmat_pthread(temp_A, local_B, local_C, m, n_threads, option, grid.my_world_rank);
       	   if (grid.my_world_rank == 0){
	   t3 = MPI_Wtime();
	   cpu_time += t3-t2;
	   }//if 
	}//else

	if (k==grid.s-1&&grid.my_world_rank==0){
		fprintf(fptr,"broadcast A[%d][%d] average take %f sec\n",m,m,network_a/grid.s);
	}//if

 	if (grid.my_world_rank==0)
           t3 = MPI_Wtime();

        MPI_Sendrecv_replace(local_B, m*m, MPI_DOUBLE,
            dest, 0, source, 0, grid.col_comm, &status);

	if (grid.my_world_rank==0){
                t4 = MPI_Wtime();
		network_b += (t4-t3);
	printf(" rank[0] broadcast B take:%f iteration:%d\n", t4-t3,k);
	if (k==grid.s-1){
		fprintf(fptr,"broadcast B[%d][%d] average take %f sec\n",m,m,network_b/grid.s);
		fprintf(fptr,"cpu_time  %f sec\n",cpu_time/grid.s);
	}//if

        }//if (grid.my_world_rank)

     }//for
}

/*===========================================================*/
double *Initialize_Matrix(PROCESS_INFO_T grid, int m) {
/*===========================================================*/

    double *temp;
    int mat_row, mat_col;
    int grid_row, grid_col;
    int dest;
    int coords[2];
    MPI_Status status;
    
    temp = (double*) malloc(m*m*sizeof(double));
    if (temp == NULL ) return temp;
    for (mat_row = 0;  mat_row < m; mat_row++) {
      for (mat_col = 0;  mat_col < m; mat_col++) {
         //temp[mat_row + mat_col*m] = 1.0;
	temp[mat_row*m + mat_col] = 1.0;
	 //modified by Hui 9_22_2011
      }//for
    }
    return temp;
}
                     

/*==================================================*/
void Set_to_zero(double*  local_A, int m  ) {
/*==================================================*/

    int i, j;
    for (i = 0; i < m*m; i++) local_A[i] = 0.0;

}



/*===============================================================================*/
void Local_matmat( double *local_A, double *local_B, double *local_C, int m) {
/*===============================================================================*/
    int i, j, k;
    static char transa = 'N';
    static double one = 1.0;

/*-------------------------------------------------------*/
/* Version 1: Just roll your own matrix-matrix multiply. */
/* This will be a *lot* slower than using dgemm.         */
/*-------------------------------------------------------*/

    for (i = 0; i < m; i++)
      for (j = 0; j < m; j++)
        for (k = 0; k < m; k++)
          local_C[i+j*m] += + local_A[i+k*m]*local_B[k+j*m];
}

void Local_matmat_pthread( double *local_A, double *local_B, double *local_C, int m,int n_threads,int option, int rank) {

int i, j;
pthread_t no_threads[n_threads+1];
void *exitstat;
ThreadsInfo threads_info[n_threads+1];

for (i=1;i<=n_threads;i++){
//threads_info[i];
threads_info[i].tid = i;
threads_info[i].m = m;
threads_info[i].local_A = local_A;
threads_info[i].local_B = local_B;
threads_info[i].local_C = local_C;
threads_info[i].n_threads = n_threads;
threads_info[i].rank = rank;

//printf("create thread:%d\n",i);
//printf("local_matmat_pthread:%d option:%d\n",i,option);

switch(option){
	case 1:
	if (pthread_create(&(no_threads[i]),NULL,doCalculate,(void *)&threads_info[i])!=0) 
	perror("Thread creation failed.\n"); 
	break;

	case 2:
	if (pthread_create(&(no_threads[i]),NULL,doTilesCalculate,(void *)&threads_info[i])!=0) 
	perror("Thread creation failed.\n"); 
	break;

	case 3:
	if (pthread_create(&(no_threads[i]),NULL,doDGEMMCalculate,(void *)&threads_info[i])!=0) 
	perror("Thread creation failed.\n"); 
	
	case 4:
	if (pthread_create(&(no_threads[i]),NULL,doCUDA,(void *)&threads_info[i])!=0)
	perror("Thread creating failed.\n");

	break;
}//switch

}//for

for (i=1; i<=n_threads; i++){
	if (pthread_join(no_threads[i],&exitstat)!=0)
	perror("joining failed");
}//for

}//local_matmat_pthread

//
//split matrix A and B into tiles each present submatrix of A and B
//this approach can optimize the program for cach, decrease the cache miss
//

void *doDGEMMCalculate(void *ptr){

int i,j,k;
double ALPHA= 1.0,BETA=0.0;

ThreadsInfo *threads_info = (ThreadsInfo *)ptr;
int m = threads_info->m;
int tid = threads_info->tid;
int n_threads = threads_info->n_threads;
int rowsperprocess = m/n_threads;
int start_row,end_row;
start_row = (tid-1)*rowsperprocess;

if (tid == n_threads)
	end_row = start_row+rowsperprocess+(m%n_threads);
else
	end_row = start_row+rowsperprocess;

//printf("m:%d,tid:%d,n_threads:%d,startpoint:%d,endpoint:%d startpoint*m:%d\n",m,tid,n_threads,startpoint,endpoint,startpoint*m);
double *A = threads_info->local_A+start_row*m; 
//double *A = &(threads_info->local_A[start_row*m]);

double *B = threads_info->local_B;
double *C = threads_info->local_C+start_row*m; 

//if (threads_info->rank == 1){
//	printf("start:%d end:%d m:%d\n",start_row,end_row,m);
//}//if

cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, end_row - start_row,m,m,
		ALPHA,A,m,B,m,BETA,C,m);
return 0;
}//doDGEMMCalculate

void *doTilesCalculate(void *ptr){
	int i,j,k;
	ThreadsInfo *threads_info = (ThreadsInfo *)ptr;
	int m = threads_info->m;
	int tid = threads_info->tid;
	int n_threads = threads_info->n_threads;

	int rowsperprocess = m/n_threads;
	int startpoint,endpoint;
	startpoint = (tid-1)*rowsperprocess;

//printf("m:%d,tid:%d,n_threads:%d,startpoint:%d,endpoint:%d\n",m,tid,n_threads,startpoint,endpoint);

	if (tid == n_threads)
		endpoint = startpoint+rowsperprocess+(m%n_threads);
	else
		endpoint = startpoint+rowsperprocess;

	//printf("m:%d,tid:%d,n_threads:%d,startpoint:%d,endpoint:%d\n",m,tid,n_threads,startpoint,endpoint);
	
	int bz = 100; //size of each tile
	
	int aHeight = endpoint - startpoint;
	int aHeightBlocks = aHeight/bz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*bz);
	if (aLastBlockHeight>0){
		aHeightBlocks++;
	}
	int bWidthBlocks = m/bz;
	int bLastBlockWidth = m - (bWidthBlocks*bz);
	if (bLastBlockWidth>0){
		bWidthBlocks++;
	}

	int commBlocks = m/bz;
	int commLastBlockWidth = m - (commBlocks*bz);
	if (commLastBlockWidth >0){
		commBlocks++;
	}//fi

	int aBlockHeight = bz;
	int bBlockWidth = bz;
	int commBlockWidth = bz;
	int ib,jb,kb;
	double *C= threads_info->local_C;
	double *A = threads_info->local_A;
	double *B = threads_info->local_B;

	for (ib=0;ib<aHeightBlocks;ib++){
		if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
			aBlockHeight = aLastBlockHeight;
		}//if

		bBlockWidth = bz;
		for (jb=0; jb<bWidthBlocks;jb++){
			if (bLastBlockWidth>0&&jb==(bWidthBlocks-1))
				bBlockWidth = bLastBlockWidth;

			commBlockWidth = bz;
			for (kb =0;kb<commBlocks;kb++){
			if (commLastBlockWidth>0 && kb==(commBlocks-1))
				commBlockWidth = commLastBlockWidth;
			for (i = startpoint+ib*bz;i<startpoint+(ib*bz)+aBlockHeight;i++){
				for (k = kb*bz;k<(kb*bz)+commBlockWidth;k++){
					for (j= jb*bz;j<(jb*bz)+bBlockWidth;j++){
					C[i*m+j]+=A[i*m+k]*B[k*m+j];
	//(threads_info->local_C[i*m+j])+= (threads_info->local_A[i*m+k])*(threads_info->local_B[k*m+j]);	
					}//for
				}
			}//for
			}//for
		}//for
	}//for
	return 0;
}

void *doCalculate(void *ptr){
  int i,j,k;

  ThreadsInfo *threads_info = (ThreadsInfo *)ptr;
  int m = threads_info->m;
  int tid = threads_info->tid;
  int n_threads = threads_info->n_threads;
  int rowsperprocess = m/n_threads;
  int startpoint,endpoint;
  startpoint = (tid -1)*rowsperprocess;
  if (tid == n_threads)
  endpoint = startpoint+rowsperprocess+(m%n_threads);
  else
  endpoint = startpoint+rowsperprocess;
  //printf("doCalculate tid = %d perprocess = %d \n",tid, rowsperprocess);
  for (i= startpoint;i<endpoint;i++)
  for (j=0;j<m;j++)
  {
  for (k=0;k<m;k++)
    threads_info->local_C[i*m+j] += threads_info->local_A[i*m+k]*threads_info->local_B[k*m+j];
  }//for
  return 0;
  //printf("doCalculate tid = %d perprocess = %d finished\n",tid, rowsperprocess);
}//doCalculate
