
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void nw_3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

    unsigned int segment = SEQUENCE_LENGTH*blockIdx.x;
    unsigned int tidx = threadIdx.x;
	int row, col, top, topleft, insertion, deletion, match, max;
    int left[COARSE_FACTOR];

	__shared__ unsigned char sequence1_s[SEQUENCE_LENGTH];
	__shared__ unsigned char sequence2_s[SEQUENCE_LENGTH];

    __shared__ int buffer2_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer3_s[SEQUENCE_LENGTH + 1];
	int * buffer2 = buffer2_s;
	int * buffer3 = buffer3_s;

    for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        unsigned int index = tidx + j*blockDim.x;
        if(index < SEQUENCE_LENGTH) {
            sequence1_s[index] = sequence1_d[segment + index];
            sequence2_s[index] = sequence2_d[segment + index];
        }
    }

	if (tidx == 0) {
		buffer2[tidx] = INSERTION;
		buffer2[tidx+1] = DELETION;
	}

    for(int i=0; i<WARP_SIZE; ++i)
	{
        if(tidx <= i)
        {
            row = i-tidx;
            col = tidx;

            topleft = (tidx==i) ? (i*DELETION) : left[0];
            top = buffer2[tidx+1];
            
            unsigned int mask = (1u << (i+1)) - 1;
            left[0] = __shfl_up_sync(mask, top, 1);

            if (tidx == 0) {
                buffer3[tidx] = (i+2)*INSERTION;
                buffer3[i+2] = (i+2)*DELETION;
                left[0] = (i+1)*INSERTION;
            }

            insertion = top + INSERTION;
            deletion = left[0] + DELETION;
            match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

            buffer3[tidx+1] = max;
        }

		int * temp = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

    __syncthreads();

	for(int i=WARP_SIZE; i<SEQUENCE_LENGTH-1; ++i)
	{
        for(unsigned int j=0; j<=i/blockDim.x; ++j)
        {
            unsigned int index = tidx + j*blockDim.x;
            if(index <= i)
            {
                row = i-index;
                col = index;

                topleft = (index==i) ? (i*DELETION) : left[j];
                top = buffer2[index+1];
                left[j] = buffer2[index];

                insertion = top + INSERTION;
                deletion = left[j] + DELETION;
                match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
                
                max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;

                buffer3[index+1] = max;
            }
        }

        if (tidx == 0) {
            buffer3[tidx] = (i+2)*INSERTION;
            buffer3[i+2] = (i+2)*DELETION;
        }

		__syncthreads();

		int * temp = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

	for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        unsigned int index = tidx + j*blockDim.x;

		row = SEQUENCE_LENGTH - 1 - index;
		col = index;

        topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left[j];
		top  = buffer2[index+1];
		left[j] = buffer2[index];

		insertion = top + INSERTION;
		deletion = left[j] + DELETION;
		match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);

		max = (insertion > deletion)?insertion:deletion;
    	max = (match > max)?match:max;

		buffer3[index] = max;
	}
	__syncthreads();

	int * temp1 = buffer2;
	buffer2 = buffer3;
	buffer3 = temp1;

    for(int j=COARSE_FACTOR-1; j>=(0); --j) //(j>=0) to be changed
    {
        unsigned int index = tidx + j*blockDim.x;
        if((index > 0)  && (index <= SEQUENCE_LENGTH-1))
        {
            row = SEQUENCE_LENGTH - index;
            col = index;

            topleft = left[j];
            top  = buffer2[index];
            left[j] = buffer2[index-1];

            insertion = top + INSERTION;
            deletion = left[j] + DELETION;
            match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

            buffer3[index] = max;
        }
    }
    __syncthreads();

    int * temp2 = buffer2;
    buffer2 = buffer3;
    buffer3 = temp2;
        
	for(int i=SEQUENCE_LENGTH-2; i>WARP_SIZE; --i)
	{   
		for(int j=COARSE_FACTOR-1; j>=(0); --j) //(j>=0) to be changed
        {
            unsigned int index = tidx + j*blockDim.x;
            if((index > SEQUENCE_LENGTH-1-i) && (index <= SEQUENCE_LENGTH-1))
            {
                row = 2*SEQUENCE_LENGTH - index - (i+1);
                col = index;
                
                topleft = left[j];
                top  = buffer2[index];
                left[j] = buffer2[index-1];

                insertion = top + INSERTION;
                deletion = left[j] + DELETION;
                match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
                
                max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;

                buffer3[index] = max;
            }
        }
        __syncthreads();

		int * temp = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

    for(int i=WARP_SIZE; i>0; --i)
	{
        unsigned int index = tidx + (COARSE_FACTOR-1)*blockDim.x;
        if((index > SEQUENCE_LENGTH-1-i) && (index <= SEQUENCE_LENGTH-1))
        {
            row = 2*SEQUENCE_LENGTH - index - (i+1);
            col = index;
            
            topleft = left[COARSE_FACTOR-1];
            top  = buffer2[index];

            /*unsigned int mask = ((1u << 32) - 1) >> i;
            left[COARSE_FACTOR-1] = __shfl_up_sync(mask, top, 1);
            if (index == SEQUENCE_LENGTH-i)
            {
                left[COARSE_FACTOR-1] = buffer2[index-1];
            }*/
            left[COARSE_FACTOR-1] = buffer2[index-1];

            insertion = top + INSERTION;
            deletion = left[COARSE_FACTOR-1] + DELETION;
            match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

            buffer3[index] = max;
        }

		int * temp = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

	if (tidx == blockDim.x -1)
	{
		scores_d[blockIdx.x] = buffer2[SEQUENCE_LENGTH-1];
	}
}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    unsigned int numThreadsPerBlock = (SEQUENCE_LENGTH+COARSE_FACTOR-1)/COARSE_FACTOR;
    unsigned int numBlocks = numSequences;
    nw_3 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d,sequence2_d,scores_d);
}
