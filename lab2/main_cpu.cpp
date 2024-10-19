#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>

#define BENCHMARK


typedef struct {
    unsigned char x, y, z, w;
} uchar4;


void roberts_method(uchar4 *in, uchar4 *out, int width, int height) {
    int x, y;
    uchar4 p;
    for(y = 0; y < height; ++y) {
        for(x = 0; x < width; ++x) {
            double w[2][2];

            // color convert
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    p = in[std::min(y+j, height-1)*width + std::min(x+i, width-1)];
                    w[i][j] = 0.299*p.x + 0.587*p.y + 0.114*p.z;
                }
            }

            double gx = w[1][1] - w[0][0];
            double gy = w[1][0] - w[0][1];
            unsigned char gf = std::min(255, int(std::sqrt(gx*gx + gy*gy)));

            uchar4 out_p = uchar4();
            out_p.x = gf;
            out_p.y = gf;
            out_p.z = gf;
            out_p.w = gf;
            out[y*width + x] = out_p;
        }
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

int main() {
    int w, h;
    uchar4 *pixels = nullptr;
    std::string input_filename, output_filename;
    std::cin >> input_filename >> output_filename;
    readData(input_filename, w, h, &pixels);
    uchar4 *pixels_out = (uchar4 *)malloc(sizeof(uchar4) * w * h);

#ifdef BENCHMARK
    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();
#endif /* BENCHMARK */

    roberts_method(pixels, pixels_out, w, h);

#ifdef BENCHMARK
    stop = std::chrono::system_clock::now();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << time << " us\n";
#endif /* BENCHMARK */

#ifndef BENCHMARK
    writeData(output_filename, w, h, pixels_out);
#endif

    free(pixels);
    free(pixels_out);
    return 0;
}
