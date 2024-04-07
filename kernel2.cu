
#include <assert.h>

#include "common.h"
#include "timer.h"
#define COARSENING 8
__global__ void nw_2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d){
    unsigned int segment = SEQUENCE_LENGTH *blockIdx.x;
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

    for(unsigned int i=0;i<COARSENING;++i){
        unsigned int index  = tidx+(i*blockDim.x);
        if(index < SEQUENCE_LENGTH){
            sequence1_s[index] = sequence1_d[segment + index];
            sequence2_s[index] = sequence2_d[segment + index];
        }
    }

	if (tidx == 0) {
		buffer1[0] = 0;
		buffer2[0] = INSERTION;
		buffer2[1] = DELETION;
	}

	__syncthreads();

	for( int i=0; i<(SEQUENCE_LENGTH-1); ++i)
	{   
        if (tidx == 0) {
			buffer3[0] = (i+2)*INSERTION;
			buffer3[i+2] = (i+2)*DELETION;
		}
		__syncthreads();
        for(unsigned int j = 0;j<=i/blockDim.x;++j){
            unsigned int index = tidx+j*blockDim.x;
            if(index <= i)
            {
                row = i-index;
                col = index;

                top = buffer2[index+1];
                left = buffer2[index];
                topleft = buffer1[index];

                insertion = top + INSERTION;
                deletion = left + DELETION;
                match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
                
                max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;

                buffer3[index+1] = max;
            }				

        }
		__syncthreads();
		int * temp = buffer1;
		buffer1 = buffer2;
		buffer2 = buffer3;
		buffer3 = temp;
	}

	for(unsigned int j=0;j<COARSENING;++j)  {
        unsigned int index = tidx+(blockDim.x*j);
		row = SEQUENCE_LENGTH - 1 - index;
		col = index;

		top  = buffer2[index+1];
		left = buffer2[index];
		topleft = buffer1[index];

		insertion = top + INSERTION;
		deletion = left + DELETION;
		match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
			
		max = (insertion > deletion)?insertion:deletion;
    	max = (match > max)?match:max;

		buffer3[index] = max;
	}
	__syncthreads();

	int * temp = buffer1;
	buffer1 = buffer2;
	buffer2 = buffer3;
	buffer3 = temp;
        
	for(int i=SEQUENCE_LENGTH-1; i>0; --i)
	{   
		for(unsigned int j = 0;j<=i/blockDim.x;++j){
            unsigned int index  = tidx+(j*blockDim.x);
            if(index < i)
            {
                row = SEQUENCE_LENGTH - index - 1;
                col = SEQUENCE_LENGTH + index - i;

                top  = buffer2[index+1];
                left = buffer2[index];
                topleft = buffer1[index+1];

                insertion = top + INSERTION;
                deletion = left + DELETION;
                match = topleft + ((sequence1_s[col] == sequence2_s[row])?MATCH:MISMATCH);
                
                max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;

                buffer3[index] = max;
            }
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

void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    unsigned int nb_threads_block = (SEQUENCE_LENGTH+COARSENING-1)/COARSENING;
    unsigned int nb_blocks = numSequences;
    nw_2<<<nb_blocks,nb_threads_block>>>(sequence1_d,sequence2_d,scores_d);

}

