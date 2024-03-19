
#ifndef _COMMON_H_
#define _COMMON_H_

#define SEQUENCE_LENGTH 1024

#define MATCH       1
#define MISMATCH    (-1)
#define INSERTION   (-1)
#define DELETION    (-1)

void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences);
void nw_gpu1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences);
void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences);
void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences);

#endif

