#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda_runtime.h>

#define BUFSIZE 256
#define NMAX 20

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void check_graphs_kernel(const char* d_batch, bool* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const char* BUFOR = &d_batch[idx * BUFSIZE];

    if (BUFOR[0] == '\0') {
        d_results[idx] = false;
        return;
    }

    int i, j, k, k3, k4, L, L1, z;
    double eps, g, h, ma, mn, norm, s, t, u, w;
    int cond;
    double d[NMAX+1], e[NMAX+1], e2[NMAX+1], Lb[NMAX+1];
    double x[NMAX+1];
    double a[NMAX*(NMAX-1)/2 + NMAX + 1];
    int n;
    int bit, poz, poz2;

    bit = 32;
    poz = 1;
    poz2 = 1;
    n = BUFOR[0] - 63;
    a[0] = 0.0;

    for (i = 0; i < n; i++)
        for (j = 0; j <= i; j++) {
            if (i == j) { a[poz2++] = 0.0; }
            else {
                if (bit == 0) { bit = 32; poz++; }
                if ((BUFOR[poz] - 63) & bit) a[poz2++] = 1.0;
                else a[poz2++] = 0.0;
                bit = bit >> 1;
            }
        }

    int k1 = 1;
    int k2 = n;
    i = 0;
    for (L = 1; L <= n; L++) { i += L; d[L] = a[i]; }

    for (L = n; L >= 2; L--) {
        i--; j = i; h = a[j]; s = 0;
        for (k = L-2; k >= 1; k--) { i--; g = a[i]; s += g*g; }
        i--;
        if (s == 0.0) { e[L] = h; e2[L] = h*h; a[j] = 0.0; }
        else {
            s += h*h; e2[L] = s; g = sqrt(s); if (h >= 0.0) g = -g;
            e[L] = g; s = 1.0 / (s - h*g); a[j] = h - g; h = 0.0; L1 = L - 1; k3 = 1;
            for (j = 1; j <= L1; j++) {
                k4 = k3; g = 0;
                for (k = 1; k <= L1; k++) {
                    g += a[k4] * a[i+k];
                    if (k < j) z = 1; else z = k;
                    k4 += z;
                }
                k3 += j; g *= s; e[j] = g; h += a[i+j] * g;
            }
            h *= 0.5 * s; k3 = 1;
            for (j = 1; j <= L1; j++) {
                s = a[i+j]; g = e[j] - h * s; e[j] = g;
                for (k = 1; k <= j; k++) { a[k3] += -s * e[k] - a[i+k] * g; k3++; }
            }
        }
        h = d[L]; d[L] = a[i+L]; a[i+L] = h;
    }
    h = d[1]; d[1] = a[1]; a[1] = h; e[1] = 0.0; e2[1] = 0.0;
    s = d[n]; t = fabs(e[n]); mn = s - t; ma = s + t;

    for (i = n-1; i >= 1; i--) {
        u = fabs(e[i]); h = t + u; t = u; s = d[i]; u = s - h;
        if (u < mn) mn = u; u = s + h; if (u > ma) ma = u;
    }
    for (i = 1; i <= n; i++) { Lb[i] = mn; x[i] = ma; }
    norm = fabs(mn); s = fabs(ma); if (s > norm) norm = s; w = ma; eps = 7.28e-17 * norm;

    for (k = k2; k >= k1; k--) {
        s = mn; i = k;
        do { cond = 0; g = Lb[i]; if (s < g) s = g; else { i--; if (i >= k1) cond = 1; } } while (cond);
        g = x[k]; if (w > g) w = g;
        while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
            if (floor(w + 10e-5) < s - 10e-5) { d_results[idx] = false; return; }
            L1 = 0; g = 1.0; t = 0.5 * (s + w);
            for (i = 1; i <= n; i++) {
                if (g != 0.0) g = e2[i] / g; else g = fabs(6.87e15 * e[i]);
                g = d[i] - t - g; if (g < 0.0) L1++;
            }
            if (L1 < k1) { s = t; Lb[k1] = s; }
            else { if (L1 < k) { s = t; Lb[L1+1] = s; if (x[L1] > t) x[L1] = t; } else w = t; }
        }
        u = 0.5 * (s + w); x[k] = u;
        if (!((ceil(u) - u < 10e-5) || (u - floor(u) < 10e-5))) { d_results[idx] = false; return; }
    }
    d_results[idx] = true;
}

int main(int argc, char *argv[]) {
    int gpu;
    CUDA_CHECK(cudaGetDeviceCount(&gpu));
    if (gpu < 1) {
        fprintf(stderr, "No GPUs are available.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaGetDevice(&gpu));
    fprintf(stderr, "Using device %d\n", gpu);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, gpu));

    int blocks = props.multiProcessorCount * 4;
    int threads;
    if (argc < 2) {
        fprintf(stderr, "Using all available GPU threads.\n");
        threads = props.maxThreadsDim[0];
    } else {
        threads = atoi(argv[1]);
        if (threads > props.maxThreadsDim[0]) {
            fprintf(stderr, "Too many threads requested, clipping to %d\n", props.maxThreadsDim[0]);
            threads = props.maxThreadsDim[0];
        }
    }

    int batchSize = blocks * threads;

    char *h_current  = (char*) calloc(batchSize, BUFSIZE);
    char *h_previous = (char*) calloc(batchSize, BUFSIZE);
    bool *h_results  = (bool*) calloc(batchSize, sizeof(bool));
    if (!h_current || !h_previous || !h_results) {
        fprintf(stderr, "Błąd alokacji pamięci hosta\n");
        exit(EXIT_FAILURE);
    }

    char *d_lines;
    bool *d_results;
    CUDA_CHECK(cudaMalloc((void**)&d_lines,   batchSize * BUFSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_results, batchSize * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_results, 0, batchSize * sizeof(bool)));

    char buffer[BUFSIZE];
    int batchIdx = 0;
    int total = 0, found = 0;
    bool firstBatch = true;

    double t_start = omp_get_wtime();

    while (fgets(buffer, BUFSIZE - 1, stdin)) {
        buffer[strcspn(buffer, "\n")] = '\0';
        memcpy(&h_current[batchIdx * BUFSIZE], buffer, BUFSIZE);
        batchIdx++;
        total++;

        if (batchIdx == batchSize) {
            if (!firstBatch) {
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_results, d_results, batchSize * sizeof(bool), cudaMemcpyDeviceToHost));
                for (int i = 0; i < batchSize; i++) {
                    if (h_results[i]) {
                        found++;
                        printf("%s\n", &h_previous[i * BUFSIZE]);
                    }
                }
            }
            firstBatch = false;

            memcpy(h_previous, h_current, batchSize * BUFSIZE);
            CUDA_CHECK(cudaMemcpy(d_lines, h_current, batchSize * BUFSIZE, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_results, 0, batchSize * sizeof(bool)));
            check_graphs_kernel<<<blocks, threads>>>(d_lines, d_results);
            CUDA_CHECK(cudaGetLastError());

            batchIdx = 0;
        }
    }

    if (batchIdx > 0) {
        for (int i = batchIdx; i < batchSize; i++) {
            h_current[i * BUFSIZE] = '\0';
        }

        if (!firstBatch) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, batchSize * sizeof(bool), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batchSize; i++) {
                if (h_results[i]) {
                    found++;
                    printf("%s\n", &h_previous[i * BUFSIZE]);
                }
            }
        }

        memcpy(h_previous, h_current, batchSize * BUFSIZE);
        CUDA_CHECK(cudaMemcpy(d_lines, h_current, batchSize * BUFSIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_results, 0, batchSize * sizeof(bool)));
        check_graphs_kernel<<<blocks, threads>>>(d_lines, d_results);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_results, d_results, batchSize * sizeof(bool), cudaMemcpyDeviceToHost));
        for (int i = 0; i < batchSize; i++) {
            if (h_results[i]) {
                found++;
                printf("%s\n", &h_previous[i * BUFSIZE]);
            }
        }
    }

    double t_end = omp_get_wtime();
    fprintf(stderr, "Czas przetwarzania: %f s\n", t_end - t_start);

    free(h_current);
    free(h_previous);
    free(h_results);
    CUDA_CHECK(cudaFree(d_lines));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}