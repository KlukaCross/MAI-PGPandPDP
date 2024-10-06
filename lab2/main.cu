#include <iostream>
#include <string>
#include <vector>

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

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    for(y = idy; y < height; y += offsety)
        for(x = idx; x < width; x += offsetx) {
            double w[2][2];

            // color convert
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    p = tex2D<uchar4>(tex, x+i, y+j);
                    w[i][j] = 0.299*p.x + 0.587*p.y + 0.114*p.z;
                }
            }

            // roberts method
            double gx = w[1][1] - w[0][0];
            double gy = w[1][0] - w[0][1];
            int gf = min(255, int(sqrt(gx*gx + gy*gy)));

            out[y*width + x] = make_uchar4(gf, gf, gf, gf);
        }
}

void readData(std::string& filename, int& w, int& h, uchar4** data) {
    FILE *f = fopen(filename.c_str(), "rb");
    fread(&w, sizeof(int), 1, f);
    fread(&h, sizeof(int), 1, f);
    *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(*data, sizeof(uchar4), w * h, f);
    fclose(f);
}

void writeData(std::string& filename, int w, int h, uchar4* data) {
    FILE *f = fopen(filename.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(data, sizeof(uchar4), w * h, f);
    fclose(f);
}

const int X_BLOCKS = 16;
const int X_THREADS = 16;
const int Y_BLOCKS = 32;
const int Y_THREADS = 32;

int main() {
    int w, h;
    uchar4 *pixels = nullptr;
    std::string input_filename, output_filename;
    std::cin >> input_filename >> output_filename;
    readData(input_filename, w, h, &pixels);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, pixels, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

#ifdef BENCHMARK
    startBenchmark();
#endif /* BENCHMARK */

    kernel<<< dim3(X_BLOCKS, X_THREADS), dim3(Y_BLOCKS, Y_THREADS) >>>(tex, dev_out, w, h);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

#ifdef BENCHMARK
    stopBenchmark();
#endif /* BENCHMARK */

    CSC(cudaMemcpy(pixels, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

#ifndef BENCHMARK
    writeData(output_filename, w, h, pixels);
#endif /* BENCHMARK */

    free(pixels);
    return 0;
}
