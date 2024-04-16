
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void nw_3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

    unsigned int segment = SEQUENCE_LENGTH*blockIdx.x;
	int row, top, topleft, insertion, deletion, match, max;
    int left[COARSE_FACTOR];
    unsigned char seq[COARSE_FACTOR];
    

	__shared__ unsigned char sequence2_s[SEQUENCE_LENGTH];

    __shared__ int buffer2_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer3_s[SEQUENCE_LENGTH + 1];
	int * buffer2 = buffer2_s;
	int * buffer3 = buffer3_s;

    #pragma unroll
    for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        unsigned int index = threadIdx.x + j*blockDim.x;
        sequence2_s[index] = sequence2_d[segment + index];
        seq[j] = sequence1_d[segment + index];
    }

	if (threadIdx.x == 0) {
		buffer2[threadIdx.x] = INSERTION;
		buffer2[threadIdx.x+1] = DELETION;
	}
    
    for(int i=0; i<WARP_SIZE; ++i)
	{
        if(threadIdx.x <= i)
        {
            row = i-threadIdx.x;

            topleft = (threadIdx.x==i) ? (i*DELETION) : left[0];
            top = buffer2[threadIdx.x+1];
            
            left[0] = __shfl_up_sync( __activemask(), top, 1);

            if (threadIdx.x == 0) {
                buffer3[threadIdx.x] = (i+2)*INSERTION;
                buffer3[i+2] = (i+2)*DELETION;
                left[0] = (i+1)*INSERTION;
            }

            insertion = top + INSERTION;
            deletion = left[0] + DELETION;
            match = topleft + ((seq[0] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;

            buffer3[threadIdx.x+1] = max;
        }

		int * temp = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}
    __syncthreads();

	for(int i=WARP_SIZE; i<SEQUENCE_LENGTH-1; ++i)
	{
        #pragma unroll
        for(unsigned int j=0; j<COARSE_FACTOR; ++j)
        {
            unsigned int index = threadIdx.x + j*blockDim.x;
            if(index <= i)
            {
                if (threadIdx.x == 0) {
                    buffer3[threadIdx.x] = (i+2)*INSERTION;
                    buffer3[i+2] = (i+2)*DELETION;
                }

                row = i-index;

                
                
                topleft = (index==i) ? (i*DELETION) : left[j];
                top = buffer2[index+1];
                left[j] = buffer2[index];
                insertion = top + INSERTION;
                deletion = left[j] + DELETION;
            

                

                match = topleft + ((seq[j] == sequence2_s[row])?MATCH:MISMATCH);
                
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
    #pragma unroll
	for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        unsigned int index = threadIdx.x + j*blockDim.x;

		row = SEQUENCE_LENGTH - 1 - index;

    
        topleft = (index == SEQUENCE_LENGTH - 1) ? (SEQUENCE_LENGTH - 1)*DELETION : left[j];
        top  = buffer2[index+1];
        left[j] = buffer2[index];
        insertion = top + INSERTION;
        deletion = left[j] + DELETION;
        

		match = topleft + ((seq[j] == sequence2_s[row])?MATCH:MISMATCH);

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
        #pragma unroll
		for(int j=COARSE_FACTOR-1; j>=0; --j) //j>=(COARSE_FACTOR-1-(i*COARSE_FACTOR/SEQUENCE_LENGTH))
        {
            unsigned int index = threadIdx.x + j*blockDim.x;
            if(index > SEQUENCE_LENGTH-1-i)  //&&(index <= SEQUENCE_LENGTH-1)
            {
                row = 2*SEQUENCE_LENGTH - index - (i+1);

            
                topleft = left[j];
                top  = buffer2[index];
                left[j] = buffer2[index-1];
                insertion = top + INSERTION;
                deletion = left[j] + DELETION;

                
                match = topleft + ((seq[j] == sequence2_s[row])?MATCH:MISMATCH);
                
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
        unsigned int index = threadIdx.x + (COARSE_FACTOR-1)*blockDim.x;
        if(index >= SEQUENCE_LENGTH-1-i)
        {
            row = 2*SEQUENCE_LENGTH - index - (i+1);
            
            topleft = left[COARSE_FACTOR-1];
            top  = max;

            left[COARSE_FACTOR-1] = __shfl_up_sync( __activemask(), top, 1);

            insertion = top + INSERTION;
            deletion = left[COARSE_FACTOR-1] + DELETION;
 
            match = topleft + ((seq[COARSE_FACTOR-1] == sequence2_s[row])?MATCH:MISMATCH);
            
            max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
        }
	}

	if (threadIdx.x == blockDim.x -1)
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
