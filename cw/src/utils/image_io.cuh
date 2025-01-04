#pragma once

#include <iostream>

void readImageData(const std::string& filename, int& w, int& h, uchar4** data) {
    FILE *f = fopen(filename.c_str(), "rb");
    fread(&w, sizeof(int), 1, f);
    fread(&h, sizeof(int), 1, f);
    *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(*data, sizeof(uchar4), w * h, f);
    fclose(f);
}

void writeImageData(const char* filename, int w, int h, uchar4* data) {
    FILE* f = fopen(filename, "wb");
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(data, sizeof(uchar4), w * h, f);
    fclose(f);
}
