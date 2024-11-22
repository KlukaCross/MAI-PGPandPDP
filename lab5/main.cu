#include <iostream>
#include <iomanip>
#include <limits>

#define CSC(call)       \
do {                    \
    cudaError_t status = call;          \
    if  (status != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));   \
        exit(0);                        \
    }                                   \
} while (0)

#define BENCHMARK

#ifdef BENCHMARK
cudaEvent_t benchmarkStart, benchmarkStop;

void startBenchmark() {
    CSC(cudaEventCreate(&benchmarkStart));
    CSC(cudaEventCreate(&benchmarkStop));
    CSC(cudaEventRecord(benchmarkStart));
}

void stopBenchmark() {
    CSC(cudaEventRecord(benchmarkStop));
    CSC(cudaEventSynchronize(benchmarkStop));
    float time;
    CSC(cudaEventElapsedTime(&time, benchmarkStart, benchmarkStop));
    CSC(cudaEventDestroy(benchmarkStart));
    CSC(cudaEventDestroy(benchmarkStop));
    std::cout << "time = " << time << " ms\n";
}
#endif /* BENCHMARK */

const int BLOCK_SIZE = 1024;

const int INT_MAX_LIMIT = std::numeric_limits<int>::max();

__device__ void swap(int* data, int first_index, int second_index) {
    int tmp = data[first_index];
    data[first_index] = data[second_index];
    data[second_index] = tmp;
}

__global__ void fillFictitious(int fictive_n, int* fictive_data) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int offset_x = gridDim.x * blockDim.x;
    for(int x = id_x; x < fictive_n; x += offset_x) {
        fictive_data[x] = INT_MAX_LIMIT;
    }
}

__global__ void preSort(int* data) {
    __shared__ int odd_even_block[BLOCK_SIZE];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    odd_even_block[threadIdx.x] = data[id_x];
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if ((threadIdx.x % 2 == 0) && (threadIdx.x < BLOCK_SIZE - 1) && (odd_even_block[threadIdx.x] > odd_even_block[threadIdx.x+1]))
            swap(odd_even_block, threadIdx.x, threadIdx.x+1);
        __syncthreads();
        if ((threadIdx.x % 2 == 1) && (threadIdx.x < BLOCK_SIZE - 1) && (odd_even_block[threadIdx.x] > odd_even_block[threadIdx.x+1]))
            swap(odd_even_block, threadIdx.x, threadIdx.x+1);
        __syncthreads();
    }

    data[id_x] = odd_even_block[threadIdx.x];
}

__global__ void bitonicMerge(int* data) {
    __shared__ int odd_even_block[2*BLOCK_SIZE];
    int first_index = 2 * threadIdx.x;
    int second_index = 2 * (BLOCK_SIZE - threadIdx.x - 1);
    odd_even_block[first_index] = data[threadIdx.x + blockIdx.x * BLOCK_SIZE];
    odd_even_block[second_index] = data[threadIdx.x + blockIdx.x * BLOCK_SIZE + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x; s > 0; s /= 2) {
        int i = threadIdx.x / s;
        int j = threadIdx.x % s;

        int index = 2 * (2 * s * i + j);
        int pair_index = index + 2*s;

        if (odd_even_block[index] > odd_even_block[pair_index])
            swap(odd_even_block, index, pair_index);
        __syncthreads();
    }

    data[threadIdx.x + blockIdx.x * BLOCK_SIZE] = odd_even_block[first_index];
    data[threadIdx.x + blockIdx.x * BLOCK_SIZE + blockDim.x] = odd_even_block[2 * (threadIdx.x + blockDim.x)];
}

void readData(int& n, int** data) {
    fread(&n, sizeof(int), 1, stdin);
    *data = (int*)malloc(sizeof(int) * n);
    fread(*data, sizeof(int), n, stdin);
}

void writeData(int n, int* data) {
    fwrite(data, sizeof(int), n, stdout);
}

int main() {
    int n;
    int* data = nullptr;
    readData(n, &data);

    int* data_dev;
    int data_dev_size = n;

    data_dev_size = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    int odd_even_number_of_blocks = data_dev_size / BLOCK_SIZE;
    CSC(cudaMalloc(&data_dev, sizeof(int) * data_dev_size));
    CSC(cudaMemcpy(data_dev, data, sizeof(int) * n, cudaMemcpyHostToDevice));

    if (n < data_dev_size) {
        fillFictitious<<<32, 32>>>(data_dev_size - n, data_dev + n);
        CSC(cudaGetLastError());
    }

    #ifdef BENCHMARK
    startBenchmark();
    #endif /* BENCHMARK */

    preSort<<<odd_even_number_of_blocks, BLOCK_SIZE>>>(data_dev);
    CSC(cudaGetLastError());

    if (odd_even_number_of_blocks > 1) {
        for (int i = 0; i < odd_even_number_of_blocks; ++i) {
            bitonicMerge<<<odd_even_number_of_blocks - 1, BLOCK_SIZE / 2>>>(data_dev + BLOCK_SIZE / 2);
            CSC(cudaGetLastError());
            bitonicMerge<<<odd_even_number_of_blocks, BLOCK_SIZE / 2>>>(data_dev);
            CSC(cudaGetLastError());
        }
    }

    #ifdef BENCHMARK
    stopBenchmark();
    #endif /* BENCHMARK */

    CSC(cudaMemcpy(data, data_dev, sizeof(int) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(data_dev));

    writeData(n, data);

    free(data);
    return 0;
}
