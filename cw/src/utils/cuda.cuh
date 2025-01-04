#pragma once

#define CSC(call)       \
do {                    \
    cudaError_t status = call;          \
    if  (status != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));   \
        exit(1);                        \
    }                                   \
} while (0)

struct cuda_timer_t {
    float ms;
    cudaEvent_t t_start, t_stop;

    cuda_timer_t() {
        CSC(cudaEventCreate(&t_start));
        CSC(cudaEventCreate(&t_stop));
    }

    void start() {
        CSC(cudaEventRecord(t_start));
    }

    void end() {
        CSC(cudaDeviceSynchronize());
        CSC(cudaGetLastError());
        CSC(cudaEventRecord(t_stop));
        CSC(cudaEventSynchronize(t_stop));
    }

    float get_time() {
        CSC(cudaEventElapsedTime(&ms, t_start, t_stop));
        return ms;
    }

    ~cuda_timer_t() {
        CSC(cudaEventDestroy(t_start));
        CSC(cudaEventDestroy(t_stop));
    }
};
