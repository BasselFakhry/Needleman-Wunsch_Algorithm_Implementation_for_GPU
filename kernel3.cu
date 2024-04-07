
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void nw_3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d) {

    

}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    unsigned int numThreadsPerBlock = (SEQUENCE_LENGTH+COARSE_FACTOR-1)/COARSE_FACTOR;
    unsigned int numBlocks = numSequences;
    nw_3 <<<numBlocks, numThreadsPerBlock>>> (sequence1_d,sequence2_d,scores_d);
}
