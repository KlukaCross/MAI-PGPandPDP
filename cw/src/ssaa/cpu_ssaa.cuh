#pragma once

void cpuSsaa(const uchar4 *data_in, uchar4 *data_out, int width, int height, int coef) {
    const double inv_coef2 = 1.0 / (coef * coef);
    const int wc = width * coef;

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
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

            data_out[j * width + i] = make_uchar4(r * inv_coef2, g * inv_coef2, b * inv_coef2, 255);
        }
    }
}

