
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void nw_1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

	unsigned int segment = SEQUENCE_LENGTH*blockIdx.x;
    unsigned int tidx = threadIdx.x;
	int row, col, top, left, topleft, insertion, deletion, match, max;

	__shared__ unsigned char sequence1_s[SEQUENCE_LENGTH];
	__shared__ unsigned char sequence2_s[SEQUENCE_LENGTH];

    __shared__ int buffer1_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer2_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer3_s[SEQUENCE_LENGTH + 1];
	int * buffer1 = buffer1_s;
	int * buffer2 = buffer2_s;
	int * buffer3 = buffer3_s;
	
	sequence1_s[tidx] = sequence1_d[segment + tidx];
	sequence2_s[tidx] = sequence2_d[segment + tidx];

	if (tidx == 0) {
		buffer1[tidx] = 0;
		buffer2[tidx] = INSERTION;
		buffer2[tidx+1] = DELETION;
	}

	__syncthreads();

	for(int i=0; i<SEQUENCE_LENGTH-1; ++i)
	{
		if (tidx == 0) {
			buffer3[tidx] = (i+2)*INSERTION;
			buffer3[i+2] = (i+2)*DELETION;
		}
		__syncthreads();
		
		if(tidx <= i)
		{
			row = i-tidx;
			col = tidx;

			top = buffer2[tidx+1];
			left = buffer2[tidx];
			topleft = buffer1[tidx];

			insertion = top + INSERTION;
			deletion = left + DELETION;
			match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
			
			max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

			buffer3[tidx+1] = max;
		}				
		__syncthreads();

		int * temp = buffer1;
		buffer1 = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

	row = SEQUENCE_LENGTH - 1 - tidx;
	col = tidx;

	top  = buffer2[tidx+1];
	left = buffer2[tidx];
	topleft = buffer1[tidx];

	insertion = top + INSERTION;
	deletion = left + DELETION;
	match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
		
	max = (insertion > deletion)?insertion:deletion;
	max = (match > max)?match:max;

	buffer3[tidx] = max;

	__syncthreads();

	int * temp = buffer1;
	buffer1 = buffer2;
	buffer2 = buffer3;
	buffer3 = temp;
        
	for(int i=SEQUENCE_LENGTH-1; i>0; --i)
	{
		if(tidx < i)
		{
			row = SEQUENCE_LENGTH - tidx - 1;
			col = SEQUENCE_LENGTH + tidx - i;

			top  = buffer2[tidx+1];
			left = buffer2[tidx];
			topleft = buffer1[tidx+1];

			insertion = top + INSERTION;
			deletion = left + DELETION;
			match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
			
			max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

			buffer3[tidx] = max;
		}
		__syncthreads();

		int * temp = buffer1;
		buffer1 = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

	if(tidx == 0)
	{
		scores_d[blockIdx.x] = buffer2[tidx];
	}
}


void nw_gpu1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {
    
    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

	unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    unsigned int numBlocks = numSequences;
	nw_1 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d, sequence2_d, scores_d);
}
