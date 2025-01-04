#pragma once

#include <string>

#include "../utils/image_io.cuh"
#include "../utils/cuda.cuh"
#include "vector3.cuh"

struct texture_t {
    int width, height;
    uchar4* data;
    uchar4* data_dev;
    bool is_use_gpu;

    __host__ __device__ texture_t() : width(0), height(0), data(nullptr), data_dev(nullptr), is_use_gpu(false) {}

    void load(const std::string& filename, bool _is_use_gpu) {
        is_use_gpu = _is_use_gpu;
        readImageData(filename.c_str(), width, height, &data);

        if (is_use_gpu) {
            CSC(cudaMalloc(&data_dev, sizeof(uchar4) * width * height));
            CSC(cudaMemcpy(data_dev, data, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));
        }
    }

    __host__ __device__ vec3f getPixel(double x, double y) const {
        int xp = static_cast<int>(x * width);
        int yp = static_cast<int>(y * height);
        xp = max(0, min(xp, width - 1));
        yp = max(0, min(yp, height - 1));
        uchar4 p;

        if (is_use_gpu) {
            p = data_dev[yp * width + xp];
        } else {
            p = data[yp * width + xp];
        }

        return vec3f(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f);
    }
};
