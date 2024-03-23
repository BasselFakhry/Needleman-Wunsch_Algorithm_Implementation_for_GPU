
#include <assert.h>

#include "common.h"
#include "timer.h"


__global__ void nw_0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {
	unsigned int tidx=threadIdx.x;
	unsigned int segment=SEQUENCE_LENGTH*blockIdx.x;
	//int matrix_base=SEQUENCE_LENGTH*SEQUENCE_LENGTH*blockIdx.x;
	__shared__ int matrix[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
	if(tidx<SEQUENCE_LENGTH){
		int row, col, top, left, topleft, insertion, deletion, match, max;

		for(int i=0;i<SEQUENCE_LENGTH;++i){

			if(tidx<=i){
				row=i-tidx;
				col=tidx;
				
				if(col==0&&row==0){
					top=col*DELETION;
					left=row*INSERTION;
					topleft=0;
				
				}		
				else if(col==0){
					top=matrix[row-1][col];
					left=row*INSERTION;
					topleft=(row-1)*INSERTION;
				
				}	
				else if(row==0){
					top=col*DELETION;
					left=matrix[row][col-1];
					topleft=(col-1)*INSERTION;
				
				}
				else{
					top=matrix[row-1][col];
					left=matrix[row][col-1];
					topleft=matrix[row-1][col-1];

				}
				
				insertion=top+INSERTION;
				deletion=left+DELETION;
				match=topleft+(sequence1_d[segment+col]==sequence2_d[segment+row])?MATCH:MISMATCH;
				max=(insertion>deletion)?insertion:deletion;
				max=(match>max)?match:max;
				matrix[row][col]=max;
			}				
			__syncthreads();
		}
		for(int i=SEQUENCE_LENGTH-1;i>0;--i){
			if(tidx<=i){
				row=i-tidx;
				col=SEQUENCE_LENGTH-i+tidx;
				top=matrix[row-1][col];
				left=matrix[row][col-1];
				topleft=matrix[row-1][col-1];
				insertion=top+INSERTION;
				deletion=left+DELETION;	
				match=topleft+(sequence1_d[segment+col]==sequence2_d[segment+row])?MATCH:MISMATCH;
				max=(insertion>deletion)?insertion:deletion;
				max=(match>max)?match:max;
				matrix[row][col]=max;
			}
			__syncthreads();
		}
	
		scores_d[blockIdx.x]= matrix[SEQUENCE_LENGTH-1][SEQUENCE_LENGTH-1];
	}
}


void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    	assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024
	//int *matrix_d;
	//cudaMalloc((void**)&matrix_d,numSequences*SEQUENCE_LENGTH*SEQUENCE_LENGTH*sizeof(int));
	int numThreadsPerBlock=1024;
    	int numEntries=numSequences*SEQUENCE_LENGTH;
    	int numBlocks=(numEntries+numThreadsPerBlock-1)/numThreadsPerBlock;
	nw_0 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d,sequence2_d,scores_d,numSequences);
}

