#include <stdio.h>

#define CSC(call)       \
do {                    \
    cudaError_t status = call;          \
    if  (status != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));   \
        exit(0);                        \
    }                                   \
} while (0)

// #define BENCHMARK


__global__ void kernel(double *v1, double *v2, double *res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        res[idx] = v1[idx] - v2[idx];
        idx += offset;
    }
}

double *readVector(int n) {
    double *v = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &v[i]);
    }
    return v;
}

void printVector(double *v, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%.10lf ", v[i]);
    }
    printf("\n");
}

const int BLOCKS = 1024;
const int THREADS = 1024;

int main() {
    int n;
    scanf("%d", &n);
    double *v1 = readVector(n);
    double *v2 = readVector(n);
    double *v_res = (double *)malloc(sizeof(double) * n);

    double *dev_v1, *dev_v2, *dev_res;
    CSC(cudaMalloc(&dev_v1, sizeof(double) * n));
    CSC(cudaMalloc(&dev_v2, sizeof(double) * n));
    CSC(cudaMalloc(&dev_res, sizeof(double) * n));
    CSC(cudaMemcpy(dev_v1, v1, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_v2, v2, sizeof(double) * n, cudaMemcpyHostToDevice));

#ifdef BENCHMARK
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));
#endif /* BENCHMARK */

    kernel<<<BLOCKS, THREADS>>>(dev_v1, dev_v2, dev_res, n);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

#ifdef BENCHMARK
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float time;
    CSC(cudaEventElapsedTime(&time, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    printf("time = %f ms\n", time);
#endif /* BENCHMARK */

    CSC(cudaMemcpy(v_res, dev_res, sizeof(double) * n, cudaMemcpyDeviceToHost));

#ifndef BENCHMARK
    printVector(v_res, n);
#endif

    free(v1);
    free(v2);
    free(v_res);
    CSC(cudaFree(dev_v1));
    CSC(cudaFree(dev_v2));
    CSC(cudaFree(dev_res));
    return 0;
}
