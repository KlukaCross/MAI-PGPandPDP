#include <iostream>
#include <cstdlib>
#include <chrono>

#define BENCHMARK


void subtraction(double *v1, double *v2, double *res, int n) {
    for (int i = 0; i < n; ++i) {
        res[i] = v1[i] - v2[i];
    }
}

double *readVector(int n) {
    double *v = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        std::cin >> v[i];
    }
    return v;
}

void printVector(double *v, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << v[i];
    }
    std::cout << "\n";
}

int main() {
    int n;
    std::cin >> n;
    double *v1 = readVector(n);
    double *v2 = readVector(n);
    double *v_res = (double *)malloc(sizeof(double) * n);

#ifdef BENCHMARK
    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();
#endif /* BENCHMARK */

    subtraction(v1, v2, v_res, n);

#ifdef BENCHMARK
    stop = std::chrono::system_clock::now();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << time << " us\n";
#endif /* BENCHMARK */

#ifndef BENCHMARK
    printVector(v_res, n);
#endif

    free(v1);
    free(v2);
    free(v_res);
    return 0;
}
