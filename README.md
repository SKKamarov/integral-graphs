# Language versions / Wersje językowe

- [English](README.md)
- [Polski](README.pl.md)

# Connected Integral Graphs of Order n = 16 and Edge Count k = 87

## 1. Abstract

The aim of this project is to implement an algorithm for selecting [integral graphs](https://en.wikipedia.org/wiki/Integral_graph) and to analyze the speedup achieved through the application of parallel techniques. The considered problem concerns connected simple graphs with a given number of vertices n=16 and edges k=87. For each graph, the eigenvalues of its adjacency matrix are computed, and their integrality is checked with a specified numerical tolerance. Only those graphs whose spectrum consists entirely of integers are printed to standard output.

The project includes three computational variants: AGS (sequential program, serving as a reference for measurements), AGOMP (CPU-parallelized version using OpenMP), and AGCUDA (GPU-accelerated implementation using CUDA technology).

The main goal of the work is to demonstrate to what extent successive levels of parallelism (multi-core CPU, massively parallel GPU) reduce graph filtering time and to identify the limitations of each architecture, including communication overhead, task granularity, and scalability.

## 2. Problem Description

An [integral graph](https://en.wikipedia.org/wiki/Integral_graph) is a simple graph whose adjacency matrix eigenvalues are all integers. Determining and classifying such graphs for a fixed order *n* and edge count *k* is an open combinatorial problem of both theoretical and practical significance. The spectra of integral graphs have applications, among others, in quantum information theory and the construction of networks with desired spectral properties.

This project investigates connected graphs with *n=16* vertices and *k=87* edges. The search space is vast, and the proportion of integral graphs within this set is very small, necessitating efficient computational methods. The generation of all non-isomorphic graphs meeting the specified parameters is handled by the external program *geng* from the *nauty* package, which outputs graphs in *graph6* format to standard output. This stream serves as input for the filtering program *AGS* (in its sequential version) or its parallel counterparts, which for each graph computes the eigenvalues of the adjacency matrix and then checks their integrality. Only graphs with an integral spectrum are passed to the output.

From a numerical computation perspective, the problem reduces to computing the spectrum of a real symmetric matrix and verifying whether each eigenvalue is an integer within a numerical tolerance. The applied method is based on Householder reduction to tridiagonal form, followed by bisection using Sturm sequences to isolate successive eigenvalues.

The choice of three algorithm variants (sequential, parallel CPU, and parallel GPU) enables an assessment of how different hardware architectures can accelerate the filtering of large graph sets.

## 3. List of Implemented Algorithms

| No. | Algorithm (Code) | Category | Purpose | Remarks |
|-----|------------------|----------|---------|---------|
| 1 | GEG | Data Generation | Generating undirected graphs with a given number of vertices and edges | Uses the nauty package |
| 2 | AGS | Sequential | Sequential search for integral graphs | Based on spectral sieve |
| 3 | AGOMP | Parallel (CPU) | Parallel search for integral graphs using multiple threads | Uses OpenMP |
| 4 | AGCUDA | Parallel (GPU) | Parallel search for integral graphs using a graphics card | Uses CUDA |

## 4. Implemented Algorithms

### 4.1. GEG - Data Generation and Measurement Program

#### 4.1.1. geng

The `geng` tool from the `nauty` package is used to generate input graphs. The command:

<code> geng -c 16 87:87 </code>

produces all non-isomorphic connected graphs with 16 vertices and 87 edges, outputting each graph on a single line in the *graph6* format. The `-c` option enforces connectivity, while the numeric parameters specify the graph order and the edge count range. Since the project also aims to investigate algorithm scaling with increasing data size, a fixed reference set containing ten million generated graphs has been prepared:

<code> geng -c 16 87:87 | head -n 10000000 > graphSet.g6 </code>

#### 4.1.2. Measurement Program

The script `computeTests` is used for performance measurements.

### 4.2. AGS - Sequential Solution

#### 4.2.1. Description

The `sito8.cu` code was used as the basis for the AGS program. All CUDA-related infrastructure was removed – GPU memory allocation, data copying to the GPU, and synchronization after kernel execution. Instead of launching a kernel on the graphics card, the computations were moved to a standard C function operating directly on a buffer in CPU memory.

It is worth noting that the original CUDA code used only one block and one thread, meaning the kernel executed sequentially, just like a regular CPU program. Removing the GPU skeleton did not change the degree of parallelization, as there was none originally.

Thanks to this modification, the AGS program no longer incurs delays related to data copying and kernel launching on the graphics card. This provides a clean, reference variant that will serve as a natural benchmark for evaluating the speedup of the OpenMP and multi-threaded CUDA versions.

For precise timing of the processing without polluting the results with I/O operations, timing markers using the `omp_get_wtime()` function were inserted into the code. They measure only the main loop processing consecutive graphs. The time is printed to standard error.

#### 4.2.2. Compilation and Execution

Compilation:

<code> gcc -fopenmp -o ags ags.c -lm </code>

Execution for a dataset with redirection to output files:

<code> ./ags < input.g6 > output.g6 2> czas.log </code>

Execution with geng:

<code> ./nauty2_8_9/geng -c 16 87:87 2>/dev/null | ./ags </code>

### 4.3. AGOMP - OpenMP

#### 4.3.1. Description

In the AGOMP program, parallelization of graph processing is achieved using OpenMP directives in a task-based model. The main thread reads consecutive lines from standard input and collects them in a local array of fixed capacity (default 1024 elements). When the array is full, the entire batch is passed to a separate task, which executes asynchronously on available threads. This task sequentially calls the analysis function for each line in the batch and frees memory after processing.

This allows the main thread to immediately continue reading and preparing the next batch of data while other threads perform computations. After the stream is exhausted, the final, incomplete batch is submitted, and the program synchronizes with all tasks. Outputting results to standard output is protected by a critical section, ensuring atomicity of each message.

Time measurement covers the entire operation period, from starting to read the first graph to finishing processing of the last line. The result is printed to stderr.

#### 4.3.2. Compilation and Execution

Compilation:

<code> gcc -fopenmp -o agomp agomp.c -lm </code>

Execution without explicitly specifying the number of threads:

<code> ./agomp < input.g6 > output.g6 2> czas.log </code>

Execution with explicit specification of thread count `N`:

<code> ./agomp N < input.g6 > output.g6 2> czas.log </code>

Execution with geng:

<code> ./geng -c 16 87:87 2>/dev/null | ./agomp </code>

### 4.4. AGCUDA - CUDA

#### 4.4.1. Description

*Section not yet available. The CUDA version implementation is currently in progress.*

#### 4.4.2. Compilation and Execution

*Section not yet available.*

## 5. Analysis

### 5.1. Analysis for Increasing Number of Graphs

The scalability of the sequential (AGS) and parallel CPU (AGOMP) implementations was examined as a function of the number of processed graphs. Measurements were taken for 14 sample sizes `N`, from 1000 to 8,192,000 lines, generated as successive powers of two multiplied by 1000. For each `N`, both programs were run five times; the table shows the minimum time obtained. The AGOMP program was run with the number of threads equal to the number of available cores, which in this case is 4. Data for the GPU-accelerated version (AGCUDA) is not yet available.

**Table 1. Computation times for AGS, AGOMP, and AGCUDA programs along with speedups for an increasing number of input graphs. S - Speedup**

| N | AGS [s] | AGOMP [s] | AGCUDA [s] | S_AGOMP | S_AGCUDA |
|---|---------|-----------|------------|---------|----------|
| 1000 | 0.018146 | 0.016122 | - | 1.13 | - |
| 2000 | 0.032739 | 0.022493 | - | 1.46 | - |
| 4000 | 0.065713 | 0.043216 | - | 1.52 | - |
| 8000 | 0.125000 | 0.076020 | - | 1.64 | - |
| 16000 | 0.259339 | 0.153810 | - | 1.69 | - |
| 32000 | 0.472716 | 0.283684 | - | 1.67 | - |
| 64000 | 1.008297 | 0.556530 | - | 1.81 | - |
| 128000 | 2.045158 | 1.087607 | - | 1.88 | - |
| 256000 | 3.973273 | 2.070440 | - | 1.92 | - |
| 512000 | 8.405879 | 3.910184 | - | 2.15 | - |
| 1024000 | 15.896943 | 8.307194 | - | 1.91 | - |
| 2048000 | 31.331593 | 17.107519 | - | 1.83 | - |
| 4096000 | 67.700554 | 34.379473 | - | 1.97 | - |
| 8192000 | 140.127044 | 68.934237 | - | 2.03 | - |

The collected results indicate that the AGS program scales in an almost linear fashion – each doubling of the number of graphs results in a proportional increase in computation time. This behavior is expected, as the algorithm analyzes each graph independently, and the cost per individual graph is similar.

The results obtained for the AGOMP program show a moderate but noticeable gain from parallelism. The speedup relative to the sequential version gradually increases from 1.13× for 1000 graphs to values oscillating around 2× for the largest tested sets (4–8 million lines). This means that the applied concurrent stream processing model allows for roughly doubling the computation speed when using eight hardware threads.

The limited scalability stems primarily from the presence of an unavoidable sequential part, which includes reading data from standard input and operations on shared buffers. While the graph analysis algorithm itself is fully parallelized, the input phase remains sequential and constitutes an increasing share of the total time for smaller samples. Additionally, with a higher number of threads, the effect of memory bus saturation increases, as all execution units access shared memory areas. Consequently, the maximum speedup does not exceed twofold and stabilizes at this level regardless of further increase in task size.

### 5.2. Analysis of the Impact of Thread Count on AGOMP Performance

To investigate thread scalability in the AGOMP program, time measurements were taken for four data sizes: 1k, 10k, 100k, and 1 million graphs. For each combination of task size and thread count (from 1 to 10), five repetitions were performed, and the table shows the minimum times. The time obtained with one thread was used as the reference point for calculating speedup (S = T₁ / Tₙ). The processor on which the measurements were performed has 4 cores x 2 threads.

**Table 2. AGOMP execution times [s] for different thread counts and speedup**

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | S₂ | S₃ | S₄ | S₅ | S₆ | S₇ | S₈ | S₉ | S₁₀ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.014746 | 0.015136 | 0.014849 | 0.015166 | 0.016572 | 0.017878 | 0.017634 | 0.017180 | 0.017396 | 0.016250 | 0.97 | 0.99 | 0.97 | 0.89 | 0.82 | 0.84 | 0.86 | 0.85 | 0.91 |
| 10000 | 0.156311 | 0.080764 | 0.080860 | 0.078457 | 0.075567 | 0.085501 | 0.083592 | 0.090033 | 0.090252 | 0.090395 | 1.94 | 1.93 | 1.99 | 2.07 | 1.83 | 1.87 | 1.74 | 1.73 | 1.73 |
| 100000 | 1.446329 | 0.802272 | 0.796545 | 0.777778 | 0.797517 | 0.770019 | 0.655603 | 0.815391 | 0.812700 | 0.635903 | 1.80 | 1.82 | 1.86 | 1.81 | 1.88 | 2.21 | 1.77 | 1.78 | 2.27 |
| 1000000 | 16.406792 | 8.230509 | 8.289438 | 8.413557 | 8.629064 | 9.633345 | 9.387560 | 9.566058 | 9.505468 | 9.302202 | 1.99 | 1.98 | 1.95 | 1.90 | 1.70 | 1.75 | 1.71 | 1.73 | 1.76 |

For the set of 1000 graphs, the processing time is practically independent of the number of threads, and in most cases is even slightly longer than with one thread. For such a small task, the overhead of creating batches and synchronizing OpenMP tasks outweighs the potential gain from parallelism.

**Table 3. Number of graphs processed per second by AGOMP**

| N | Best Time [s] | Graphs/s |
|---|--------------------|---------|
| 1000 | 0.014746 | 67,815 |
| 10000 | 0.075567 | 132,333 |
| 100000 | 0.635903 | 157,257 |
| 1000000 | 8.230509 | 121,499 |

The highest throughput was achieved for the set of 100,000 graphs – approximately 157,000 graphs per second. For the largest dataset, throughput decreases slightly, which may be due to caching effects and memory management.

### 5.3. Correctness Verification Summary

*Section not yet available. A comparison of the results from AGS, AGOMP, and AGCUDA with correctness verification will be completed after the CUDA implementation is finished.*
