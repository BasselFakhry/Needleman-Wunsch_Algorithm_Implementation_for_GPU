
#include <assert.h>

#include "common.h"
#include "timer.h"

#define COARSE_FACTOR 4

__global__ void nw_3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

    int index, match, max;
    int* temp;
    int left[COARSE_FACTOR];
    int seq1[COARSE_FACTOR];
	__shared__ unsigned char sequence2_s[SEQUENCE_LENGTH];

    __shared__ int buffer1_s[SEQUENCE_LENGTH + 1];
    __shared__ int buffer2_s[SEQUENCE_LENGTH + 1];
	int * buffer1 = buffer1_s;
	int * buffer2 = buffer2_s;

    #pragma unroll
    for(unsigned int j=0; j<COARSE_FACTOR; ++j)
    {
        index = threadIdx.x + j*blockDim.x;
        seq1[j] = sequence1_d[SEQUENCE_LENGTH*blockIdx.x + index];
        sequence2_s[index] = sequence2_d[SEQUENCE_LENGTH*blockIdx.x + index];
        left[j] = index*DELETION;
        buffer1[index+1] = (index+1)*DELETION;
	    buffer2[index+1] = (index+1)*DELETION;
    }

    #pragma unroll
    for(int i=0; i<WARP_SIZE; ++i)
	{
        if(threadIdx.x<=i)
        {
            match = left[0];
            left[0] = __shfl_up_sync( __activemask(), buffer1[threadIdx.x+1], 1);

            if (threadIdx.x == 0) {
                buffer2[0] = (i+2)*INSERTION;
                left[0] = (i+1)*INSERTION;
            }

            match += ((seq1[0] == sequence2_s[i-threadIdx.x])?MATCH:MISMATCH);
            max = (buffer1[threadIdx.x+1]+INSERTION>left[0]+DELETION)?(buffer1[threadIdx.x+1]+INSERTION):(left[0]+DELETION);
            buffer2[threadIdx.x+1] = (match>max)?match:max;
        }

		temp = buffer1;
		buffer1 = buffer2;
		buffer2 = temp;
	}
    __syncthreads();

    #pragma unroll
	for(int i=WARP_SIZE; i<SEQUENCE_LENGTH-64; ++i)
	{
        #pragma unroll
        for(unsigned int j=0; j<COARSE_FACTOR-2; ++j)
        {
            index = threadIdx.x + j*blockDim.x;
            if(index>i) {
                break;
            } else {
                match = left[j];
                left[j] = buffer1[index];
                
                match += ((seq1[j] == sequence2_s[i-index])?MATCH:MISMATCH);
                max = (buffer1[index+1]+INSERTION>left[j]+DELETION)?(buffer1[index+1]+INSERTION):(left[j]+DELETION);
                buffer2[index+1] = (match>max)?match:max;
            }
        }
		__syncthreads();

		temp = buffer1;
		buffer1 = buffer2;
		buffer2 = temp;
	}
    
    #pragma unroll
	for(int i=SEQUENCE_LENGTH-64; i<SEQUENCE_LENGTH-1; ++i)
	{
        #pragma unroll
        for(unsigned int j=1; j<COARSE_FACTOR-1; ++j)
        {
            index = threadIdx.x + j*blockDim.x;
            if(index>i) {
                break;
            } else {
                match = left[j];
                left[j] = buffer1[index];
                
                match += ((seq1[j] == sequence2_s[i-index])?MATCH:MISMATCH);
                max = (buffer1[index+1]+INSERTION>left[j]+DELETION)?(buffer1[index+1]+INSERTION):(left[j]+DELETION);
                buffer2[index+1] = (match>max)?match:max;
            }
        }
		__syncthreads();

		temp = buffer1;
		buffer1 = buffer2;
		buffer2 = temp;
	}

    #pragma unroll
	for(unsigned int j=1; j<COARSE_FACTOR-1; ++j)
    {
        index = threadIdx.x + j*blockDim.x;
        match = left[j];
        left[j] = buffer1[index];

		match += ((seq1[j] == sequence2_s[SEQUENCE_LENGTH-1-index])?MATCH:MISMATCH);
		max = (buffer1[index+1]+INSERTION>left[j]+DELETION)?(buffer1[index+1]+INSERTION):(left[j]+DELETION);
    	buffer2[index] = (match>max)?match:max;
	}
	__syncthreads();

	temp = buffer1;
	buffer1 = buffer2;
	buffer2 = temp;
    
    #pragma unroll
	for(int i=SEQUENCE_LENGTH-1; i>SEQUENCE_LENGTH-65; --i)
	{
        #pragma unroll
		for(int j=COARSE_FACTOR-2; j>0; --j)
        {
            index = threadIdx.x + j*blockDim.x;
            if(index<SEQUENCE_LENGTH-i) {
                break;
            } else {
                match = left[j];
                left[j] = buffer1[index-1];

                match += ((seq1[j] == sequence2_s[2*SEQUENCE_LENGTH-index-(i+1)])?MATCH:MISMATCH);
                max = (buffer1[index]+INSERTION>left[j]+DELETION)?(buffer1[index]+INSERTION):(left[j]+DELETION);
                max = (match>max)?match:max;
                buffer2[index] = max;
            }
        }
        __syncthreads();

		temp = buffer1;
		buffer1 = buffer2;
		buffer2 = temp;
	}

    #pragma unroll
	for(int i=SEQUENCE_LENGTH-65; i>0; --i)
	{
        #pragma unroll
		for(int j=COARSE_FACTOR-1; j>1; --j)
        {
            index = threadIdx.x + j*blockDim.x;
            if(index<SEQUENCE_LENGTH-i) {
                break;
            } else {
                match = left[j];
                left[j] = buffer1[index-1];

                match += ((seq1[j] == sequence2_s[2*SEQUENCE_LENGTH-index-(i+1)])?MATCH:MISMATCH);
                max = (buffer1[index]+INSERTION>left[j]+DELETION)?(buffer1[index]+INSERTION):(left[j]+DELETION);
                max = (match>max)?match:max;
                buffer2[index] = max;
            }
        }
        __syncthreads();

		temp = buffer1;
		buffer1 = buffer2;
		buffer2 = temp;
	}

	if (threadIdx.x == blockDim.x-1) {
		scores_d[blockIdx.x] = max;
	}
}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    unsigned int numThreadsPerBlock = SEQUENCE_LENGTH/COARSE_FACTOR;
    unsigned int numBlocks = numSequences;
    nw_3 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d,sequence2_d,scores_d);
}