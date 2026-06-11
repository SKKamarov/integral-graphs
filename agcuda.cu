#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NMAX 20
#define MAX_LINE 22

// Makro do sprawdzania błędów CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Jądro GPU, każdy wątek przetwarza jeden graf
__global__ void check_graphs_kernel(const char* d_batch, int* d_results, long long num_graphs) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_graphs) return;

    const char* BUFOR = &d_batch[idx * MAX_LINE];
    d_results[idx] = 0;

    int i, j, k, k3, k4, L, L1, z;
    double eps, g, h, ma, mn, norm, s, t, u, w;
    int cond;
    double d[NMAX+1], e[NMAX+1], e2[NMAX+1], Lb[NMAX+1];
    double x[NMAX+1];
    double a[NMAX*(NMAX-1)/2 + NMAX + 1];
    int n;

    int bit = 32, poz = 1, poz2 = 1;
    n = BUFOR[0] - 63;
    a[0] = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            if (i == j) { a[poz2++] = 0.0; }
            else {
                if (bit == 0) { bit = 32; poz++; }
                if ((BUFOR[poz] - 63) & bit) { a[poz2++] = 1.0; }
                else { a[poz2++] = 0.0; }
                bit = bit >> 1;
            }
        }
    }

    int k1 = 1, k2 = n;
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
    h = d[1]; d[1] = a[1]; a[1] = h; e[1] = 0.0; e2[1] = 0.0; s = d[n];
    t = fabs(e[n]); mn = s - t; ma = s + t;

    for (i = n-1; i >= 1; i--) {
        u = fabs(e[i]); h = t + u; t = u; s = d[i]; u = s - h;
        if (u < mn) mn = u; u = s + h; if (u > ma) ma = u;
    }
    for (i = 1; i <= n; i++) { Lb[i] = mn; x[i] = ma; }
    norm = fabs(mn); s = fabs(ma); if (s > norm) norm = s; w = ma; eps = 7.28e-17 * norm;

    for (k = k2; k >= k1; k--) {
        s = mn; i = k;
        do {
            cond = 0; g = Lb[i];
            if (s < g) s = g;
            else { i--; if (i >= k1) cond = 1; }
        } while (cond);
        g = x[k]; if (w > g) w = g;
        while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
            if (floor(w + 10e-5) < s - 10e-5) return;
            L1 = 0; g = 1.0; t = 0.5 * (s + w);
            for (i = 1; i <= n; i++) {
                if (g != 0.0) g = e2[i] / g;
                else          g = fabs(6.87e15 * e[i]);
                g = d[i] - t - g;
                if (g < 0.0) L1++;
            }
            if (L1 < k1) { s = t; Lb[k1] = s; }
            else { if (L1 < k) { s = t; Lb[L1+1] = s; if (x[L1] > t) x[L1] = t; } else w = t; }
        }
        u = 0.5 * (s + w); x[k] = u;
        if (!((ceil(u) - u < 10e-5) || (u - floor(u) < 10e-5))) {
            return;
        }
    }

    // Jeśli dotarliśmy tutaj – graf jest całkowity
    d_results[idx] = 1;
}

int main() {
    char **h_lines = NULL;
    long long capacity = 100000;
    long long graph_count = 0;

    h_lines = (char**) malloc(capacity * sizeof(char*));
    if (!h_lines) return 1;

    char buffer[1024];

    // Wczytanie wszystkich linii do pamięci
    while (fgets(buffer, sizeof(buffer), stdin)) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n')
            buffer[len-1] = '\0';

        if (graph_count >= capacity) {
            capacity *= 2;
            h_lines = (char**) realloc(h_lines, capacity * sizeof(char*));
            if (!h_lines) return 1;
        }
        h_lines[graph_count] = strdup(buffer);
        graph_count++;
    }

    if (graph_count == 0) {
        free(h_lines);
        return 0;
    }

    // Przygotowanie spłaszczonej tablicy na CPU (stała długość rekordu)
    char *h_batch = (char*) malloc(graph_count * MAX_LINE);
    memset(h_batch, 0, graph_count * MAX_LINE);
    for (long long i = 0; i < graph_count; i++) {
        strcpy(&h_batch[i * MAX_LINE], h_lines[i]);
        free(h_lines[i]);
    }
    free(h_lines);

    // Alokacja na GPU
    char *d_batch;
    int *d_results;
    int *h_results = (int*) malloc(graph_count * sizeof(int));

    CUDA_CHECK(cudaMalloc((void**)&d_batch, graph_count * MAX_LINE));
    CUDA_CHECK(cudaMalloc((void**)&d_results, graph_count * sizeof(int)));

    // Start pomiaru czasu
    double t_start = omp_get_wtime();

    // Kopiowanie danych na GPU
    CUDA_CHECK(cudaMemcpy(d_batch, h_batch, graph_count * MAX_LINE, cudaMemcpyHostToDevice));

    // Konfiguracja uruchomienia
    int threadsPerBlock = 256;
    long long blocksPerGrid = (graph_count + threadsPerBlock - 1) / threadsPerBlock;

    // Uruchomienie jądra
    check_graphs_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_batch, d_results, graph_count);

    // Synchronizacja i pobranie wyników
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, graph_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Koniec pomiaru czasu
    double t_end = omp_get_wtime();
    fprintf(stderr, "Czas przetwarzania: %f s\n", t_end - t_start);

    // Zapis grafów całkowitych na stdout
    for (long long i = 0; i < graph_count; i++) {
        if (h_results[i] == 1) {
            printf("%s\n", &h_batch[i * MAX_LINE]);
        }
    }

    // Sprzątanie
    cudaFree(d_batch);
    cudaFree(d_results);
    free(h_batch);
    free(h_results);

    return 0;
}