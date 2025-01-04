#pragma once

__global__ void gpuSsaa(const uchar4 *data_in, uchar4 *data_out, int w, int h, int coef) {
    const double inv_coef2 = 1.0 / (coef * coef);
    const int wc = w * coef;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            double r = 0, g = 0, b = 0;
            int baseIdx = (coef * j) * wc + coef * i;

            for (int ki = 0; ki < coef; ++ki) {
                for (int kj = 0; kj < coef; ++kj) {
                    uchar4 pix = data_in[baseIdx + kj * wc + ki];
                    r += pix.x;
                    g += pix.y;
                    b += pix.z;
                }
            }

            data_out[j * w + i] = make_uchar4(r * inv_coef2, g * inv_coef2, b * inv_coef2, 255);
        }
    }
}
