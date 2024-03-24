
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "timer.h"

void nw_cpu(unsigned char* sequence1, unsigned char* sequence2, int* scores, unsigned int numSequences) {
    for(unsigned int s = 0; s < numSequences; ++s) {
        int matrix[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        for(int i2 = 0; i2 < SEQUENCE_LENGTH; ++i2) {
            for (int i1 = 0; i1 < SEQUENCE_LENGTH; ++i1) {
                // Get neighbors
                int top     = (i2 == 0)?((i1 + 1)*DELETION):(matrix[i2 - 1][i1]);
                int left    = (i1 == 0)?((i2 + 1)*INSERTION):(matrix[i2][i1 - 1]);
                int topleft = (i2 == 0)?(i1*DELETION):((i1 == 0)?(i2*INSERTION):(matrix[i2 - 1][i1 - 1]));
                // Find scores based on neighbors
                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match     = topleft + ((sequence2[s*SEQUENCE_LENGTH + i2] == sequence1[s*SEQUENCE_LENGTH + i1])?MATCH:MISMATCH);
                // Select best score
                int max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;
                matrix[i2][i1] = max;
            }
        }
        scores[s] = matrix[SEQUENCE_LENGTH - 1][SEQUENCE_LENGTH - 1];
    }
}

void verify(int* scores_cpu, int* scores_gpu, unsigned int numSequences) {
    for(unsigned int s = 0; s < numSequences; ++s) {
        if(scores_cpu[s] != scores_gpu[s]) {
            printf("\033[1;31mMismatch at sequence s = %u (CPU result = %d, GPU result = %d)\033[0m\n", s, scores_cpu[s], scores_gpu[s]);
            return;
        } else { printf("score = %d\n", scores_cpu[s]); // XXX
        }
    }
    printf("Verification succeeded\n");
}

void mutateSequence(unsigned char* sequence1, unsigned char* sequence2) {
    const float PROB_MATCH = 0.80f;
    const float PROB_INS   = 0.10f;
    const float PROB_DEL   = 1.00f - PROB_MATCH - PROB_INS;
    assert(PROB_MATCH >= 0.00f && PROB_MATCH <= 1.00f);
    assert(PROB_INS   >= 0.00f && PROB_INS   <= 1.00f);
    assert(PROB_DEL   >= 0.00f && PROB_DEL   <= 1.00f);
    unsigned int i1 = 0, i2 = 0;
    while(i1 < SEQUENCE_LENGTH && i2 < SEQUENCE_LENGTH) {
        float prob = rand()*1.0f/RAND_MAX;
        if(prob < PROB_MATCH) {
            sequence2[i2++] = sequence1[i1++]; // Match
        } else if(prob < PROB_MATCH + PROB_INS) {
            sequence2[i2++] = rand()%256; // Insertion
        } else {
            ++i1; // Deletion
        }
    }
    while(i2 < SEQUENCE_LENGTH) {
        sequence2[i2++] = rand()%256; // Tail insertions
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Parse arguments
    unsigned int numSequences = 3000;
    unsigned int runGPUVersion0 = 0;
    unsigned int runGPUVersion1 = 0;
    unsigned int runGPUVersion2 = 0;
    unsigned int runGPUVersion3 = 0;
    int opt;
    while((opt = getopt(argc, argv, "N:0123")) >= 0) {
        switch(opt) {
            case 'N': numSequences = atoi(optarg);  break;
            case '0': runGPUVersion0 = 1;           break;
            case '1': runGPUVersion1 = 1;           break;
            case '2': runGPUVersion2 = 1;           break;
            case '3': runGPUVersion3 = 1;           break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // Allocate memory and initialize data
    printf("Initializing %u sequence pairs of length %u per pair\n", numSequences, SEQUENCE_LENGTH);
    Timer timer;
    unsigned char* sequence1 = (unsigned char*) malloc(numSequences*SEQUENCE_LENGTH*sizeof(unsigned char));
    unsigned char* sequence2 = (unsigned char*) malloc(numSequences*SEQUENCE_LENGTH*sizeof(unsigned char));
    int* scores_cpu = (int*) malloc(numSequences*sizeof(int));
    int* scores_gpu = (int*) malloc(numSequences*sizeof(int));
    for(unsigned int s = 0; s < numSequences; ++s) {
        for(unsigned int i = 0; i < SEQUENCE_LENGTH; ++i) {
            sequence1[s*SEQUENCE_LENGTH + i] = rand()%256;
        }
        mutateSequence(&sequence1[s*SEQUENCE_LENGTH], &sequence2[s*SEQUENCE_LENGTH]);
    }

    // Compute on CPU
    startTime(&timer);
    nw_cpu(sequence1, sequence2, scores_cpu, numSequences);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    if(runGPUVersion0 || runGPUVersion1 || runGPUVersion2 || runGPUVersion3) {

        // Allocate GPU memory
        startTime(&timer);
        unsigned char *sequence1_d;
        unsigned char *sequence2_d;
        int *scores_d;
        cudaMalloc((void**) &sequence1_d, numSequences*SEQUENCE_LENGTH*sizeof(unsigned char));
        cudaMalloc((void**) &sequence2_d, numSequences*SEQUENCE_LENGTH*sizeof(unsigned char));
        cudaMalloc((void**) &scores_d, numSequences*sizeof(int));
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Allocation time");

        // Copy data to GPU
        startTime(&timer);
        cudaMemcpy(sequence1_d, sequence1, numSequences*SEQUENCE_LENGTH*sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(sequence2_d, sequence2, numSequences*SEQUENCE_LENGTH*sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Copy to GPU time");

        if(runGPUVersion0) {

            // Reset
            cudaMemset(scores_d, 0, numSequences*sizeof(int));
            cudaDeviceSynchronize();

            // Compute on GPU with version 0
            startTime(&timer);
            nw_gpu0(sequence1_d, sequence2_d, scores_d, numSequences);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "GPU kernel time (version 0)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(scores_gpu, scores_d, numSequences*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(scores_cpu, scores_gpu, numSequences);

        }

        if(runGPUVersion1) {

            // Reset
            cudaMemset(scores_d, 0, numSequences*sizeof(int));
            cudaDeviceSynchronize();

            // Compute on GPU with version 1
            startTime(&timer);
            nw_gpu1(sequence1_d, sequence2_d, scores_d, numSequences);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "GPU kernel time (version 1)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(scores_gpu, scores_d, numSequences*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(scores_cpu, scores_gpu, numSequences);

        }

        if(runGPUVersion2) {

            // Reset
            cudaMemset(scores_d, 0, numSequences*sizeof(int));
            cudaDeviceSynchronize();

            // Compute on GPU with version 2
            startTime(&timer);
            nw_gpu2(sequence1_d, sequence2_d, scores_d, numSequences);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "GPU kernel time (version 2)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(scores_gpu, scores_d, numSequences*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(scores_cpu, scores_gpu, numSequences);

        }

        if(runGPUVersion3) {


            // Reset
            cudaMemset(scores_d, 0, numSequences*sizeof(int));
            cudaDeviceSynchronize();

            // Compute on GPU with version 3
            startTime(&timer);
            nw_gpu3(sequence1_d, sequence2_d, scores_d, numSequences);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "GPU kernel time (version 3)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(scores_gpu, scores_d, numSequences*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(scores_cpu, scores_gpu, numSequences);

        }

        // Free GPU memory
        startTime(&timer);
        cudaFree(sequence1_d);
        cudaFree(sequence2_d);
        cudaFree(scores_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Deallocation time");

    }

    // Free memory
    free(sequence1);
    free(sequence2);
    free(scores_cpu);
    free(scores_gpu);

    return 0;

}

