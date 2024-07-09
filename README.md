# Needleman-Wunsch Algorithm Implementation for GPU

This project demonstrates the implementation of the Needleman-Wunsch algorithm for exact string matching, developed as part of the CMPS224 GPU Computing course at the American University of Beirut (AUB).

## Overview

The Needleman-Wunsch algorithm is a dynamic programming technique widely used in bioinformatics for aligning protein or nucleotide sequences to identify similarities, differences, and evolutionary relationships. Our primary objective was to leverage parallel computing on GPUs to significantly enhance the algorithm's performance.

## Achievements

- **Speedup:** Achieved an exceptional speedup of approximately 3800x compared to the sequential CPU implementation.
- **Execution Time:** Reduced execution time to an impressive 4.7ms, outperforming the professor's implementation which had a runtime of 5.8ms.
- **Recognition:** Earned a perfect score of 100 on the project for achieving the fastest implementation in the class.

## Applied Optimizations

Incremental optimizations were applied, with each kernel building on the previous one.

Hardware used: NVIDIA V100 GPU.

- **Kernel0:**
  - Used an array of matrices, each storing calculations for a pair of sequences.
  - Each block handled one sequence pair, with the number of threads per block equaling SEQUENCE_LENGTH, traversing the matrix anti-diagonally and assigning a thread to each output element in the current diagonal.
  - Non-coalesced accesses to global memory for sequence pairs and matrix operations.
  - Resulted in a large allocation and an execution time of 2181ms for the default 3000 sequence pairs.

- **Kernel1:**
  - Stored sequence pairs in shared memory to reduce global memory accesses.
  - Replaced the entire matrix with three buffers in shared memory for the current diagonals.
  - Achieved coalesced accesses, improving memory efficiency.
  - Reduced execution time to 14.57ms for the default 3000 sequence pairs.

- **Kernel2:**
  - Implemented thread coarsening, reducing the number of threads per block and minimizing idle threads (coarse factor 2).
  - Further reduced execution time to 11.48ms for the default 3000 sequence pairs.

- **Kernel3:**
  - Modified indexing so each thread handled one column in the matrix, enabling the use of registers and eliminating the need for both buffer 3 and the seq1 buffer in shared memory.
  - Allowed 8 blocks to run simultaneously on one SM, aided by the -maxrregcount=32 flag (coarse factor 4).
  - Applied #pragma unroll to reduce stalls on long-latency operations.
  - Minimized control divergence and unnecessary computations with optimized loops and break conditions.
  - Achieved the final execution time of 4.7ms for the default 3000 sequence pairs.


## Instructions

To compile:

```
make
```

To run:

```
./nw [flags]

```

Optional flags:

```
  -N <N>    the number of sequence pairs to match

  -0        run GPU version 0
  -1        run GPU version 1
  -2        run GPU version 2
  -3        run GPU version 3
            NOTE: It is okay to specify multiple different GPU versions in the
                  same run. By default, only the CPU version is run.
```

## License

© 2024 CMPS224_GroupDevin. All Rights Reserved.