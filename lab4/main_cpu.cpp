#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <math.h>

#define BENCHMARK


void swapRows(double* matrix, int n, int row1, int row2) {
    for (int x = 0; x < n; x += 1) {
        double tmp = matrix[x*n + row1];
        matrix[x*n + row1] = matrix[x*n + row2];
        matrix[x*n + row2] = tmp;
    }
}

void updateMatrix(double* matrix, int n, int i) {
    for (int x = i+1; x < n; x += 1)
        for (int y = i+1; y < n; y += 1)
            matrix[y*n + x] += matrix[y*n + i] * (-matrix[i*n + x] / matrix[i*n + i]);
}

double calculateDet(double* matrix, int n, bool is_even_swaps) {
    double res = 1;
    for (int i = 0; i < n; ++i) {
        if (fabs(matrix[i*n + i]) < 1e-7)
            return 0;
        res *= matrix[i*n + i];
    }
    if (is_even_swaps)
        res *= -1;
    return res;
}

int main() {
    int n;
    std::cin >> n;
    double* matrix = (double*)malloc(sizeof(double) * n*n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            std::cin >> matrix[j*n + i];

    bool is_even_swaps = false;

#ifdef BENCHMARK
    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();
#endif /* BENCHMARK */

    for (int i = 0; i < n-1; ++i) {
        int max_row = i;
        double max_number = matrix[i*n+i];
        for (int j = i+1; j < n; ++j) {
            if (max_number < matrix[i*n + j]) {
                max_number = matrix[i*n + j];
                max_row = j;
            }
        }
        if (max_row != i) {
            swapRows(matrix, n, i, max_row);
            is_even_swaps ^= true;
        }

        updateMatrix(matrix, n, i);
    }

#ifdef BENCHMARK
    stop = std::chrono::system_clock::now();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << time << " us\n";
#endif /* BENCHMARK */

    double result = calculateDet(matrix, n, is_even_swaps);
    std::cout<< std::scientific << std::setprecision(10) << result;

    free(matrix);
    return 0;
}
