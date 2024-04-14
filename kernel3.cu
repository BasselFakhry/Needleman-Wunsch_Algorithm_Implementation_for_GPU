
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void nw_3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

    unsigned int segment = SEQUENCE_LENGTH*blockIdx.x;
    unsigned int tidx = threadIdx.x;
	int row, col, top, topleft, insertion, deletion, match, max;
    int left0, left1, left2, left3;

	__shared__ unsigned char sequence1_s[SEQUENCE_LENGTH];
	__shared__ unsigned char sequence2_s[SEQUENCE_LENGTH];

    __shared__ int buffer2_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer3_s[SEQUENCE_LENGTH + 1];
	int * buffer2 = buffer2_s;
	int * buffer3 = buffer3_s;

    for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        unsigned int index = tidx + j*blockDim.x;
        sequence1_s[index] = sequence1_d[segment + index];
        sequence2_s[index] = sequence2_d[segment + index];
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

            topleft = (tidx==i) ? (i*DELETION) : left0;
            top = buffer2[tidx+1];
            
            left0 = __shfl_up_sync( __activemask(), top, 1);

            if (tidx == 0) {
                buffer3[tidx] = (i+2)*INSERTION;
                buffer3[i+2] = (i+2)*DELETION;
                left0 = (i+1)*INSERTION;
            }

            insertion = top + INSERTION;
            deletion = left0 + DELETION;
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
                if (tidx == 0) {
                    buffer3[tidx] = (i+2)*INSERTION;
                    buffer3[i+2] = (i+2)*DELETION;
                }

                row = i-index;
                col = index;

                if (j==0) {
                    topleft = (index==i) ? (i*DELETION) : left0;
                    top = buffer2[index+1];
                    left0 = buffer2[index];
                    insertion = top + INSERTION;
                    deletion = left0 + DELETION;

                } else if (j==1) {
                    topleft = (index==i) ? (i*DELETION) : left1;
                    top = buffer2[index+1];
                    left1 = buffer2[index];
                    insertion = top + INSERTION;
                    deletion = left1 + DELETION;
                    
                } else if (j==2) {
                    topleft = (index==i) ? (i*DELETION) : left2;
                    top = buffer2[index+1];
                    left2 = buffer2[index];
                    insertion = top + INSERTION;
                    deletion = left2 + DELETION;

                } else if (j==3) {
                    topleft = (index==i) ? (i*DELETION) : left3;
                    top = buffer2[index+1];
                    left3 = buffer2[index];
                    insertion = top + INSERTION;
                    deletion = left3 + DELETION;
                }

                match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
                
                max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;

                buffer3[index+1] = max;
            }
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

        if (j==0) {
            topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left0;
            top  = buffer2[index+1];
            left0 = buffer2[index];
            insertion = top + INSERTION;
            deletion = left0 + DELETION;

        } else if (j==1) {
            topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left1;
            top  = buffer2[index+1];
            left1 = buffer2[index];
            insertion = top + INSERTION;
            deletion = left1 + DELETION;
            
        } else if (j==2) {
            topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left2;
            top  = buffer2[index+1];
            left2 = buffer2[index];
            insertion = top + INSERTION;
            deletion = left2 + DELETION;

        } else if (j==3) {
            topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left3;
            top  = buffer2[index+1];
            left3 = buffer2[index];
            insertion = top + INSERTION;
            deletion = left3 + DELETION;
        }

		match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);

		max = (insertion > deletion)?insertion:deletion;
    	max = (match > max)?match:max;

		buffer3[index] = max;
	}
	__syncthreads();

	int * temp = buffer2;
	buffer2 = buffer3;
	buffer3 = temp;
        
	for(int i=SEQUENCE_LENGTH-1; i>=WARP_SIZE; --i)
	{
		for(int j=COARSE_FACTOR-1; j>=0; --j) //j>=(COARSE_FACTOR-1-(i*COARSE_FACTOR/SEQUENCE_LENGTH))
        {
            unsigned int index = tidx + j*blockDim.x;
            if(index > SEQUENCE_LENGTH-1-i)  //&&(index <= SEQUENCE_LENGTH-1)
            {
                row = 2*SEQUENCE_LENGTH - index - (i+1);
                col = index;

                if (j==0) {
                    topleft = left0;
                    top  = buffer2[index];
                    left0 = buffer2[index-1];
                    insertion = top + INSERTION;
                    deletion = left0 + DELETION;

                } else if (j==1) {
                    topleft = left1;
                    top  = buffer2[index];
                    left1 = buffer2[index-1];
                    insertion = top + INSERTION;
                    deletion = left1 + DELETION;
                    
                } else if (j==2) {
                    topleft = left2;
                    top  = buffer2[index];
                    left2 = buffer2[index-1];
                    insertion = top + INSERTION;
                    deletion = left2 + DELETION;

                } else if (j==3) {
                    topleft = left3;
                    top  = buffer2[index];
                    left3 = buffer2[index-1];
                    insertion = top + INSERTION;
                    deletion = left3 + DELETION;
                }

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

    for(int i=WARP_SIZE-1; i>0; --i)
	{
        unsigned int index = tidx + (COARSE_FACTOR-1)*blockDim.x;
        if(index >= SEQUENCE_LENGTH-1-i)
        {
            row = 2*SEQUENCE_LENGTH - index - (i+1);
            col = index;
            
            topleft = left3;
            top  = max;

            left3 = __shfl_up_sync( __activemask(), top, 1);

            insertion = top + INSERTION;
            deletion = left3 + DELETION;
 
            match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
        }
	}

	if (tidx == blockDim.x -1)
	{
		scores_d[blockIdx.x] = max;
	}
}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    unsigned int numThreadsPerBlock = SEQUENCE_LENGTH/COARSE_FACTOR;
    unsigned int numBlocks = numSequences;
    nw_3 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d,sequence2_d,scores_d);
}
