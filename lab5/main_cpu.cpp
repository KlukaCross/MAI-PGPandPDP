#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <math.h>
#include <vector>
#include <unistd.h>

#define BENCHMARK

void oddEvenSort(std::vector<int>& arr) {
    bool isSorted = false;
    int n = arr.size();

    while (!isSorted) {
        isSorted = true;

        for (int i = 0; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

        for (int i = 1; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

int main() {
    int n;

    if (read(STDIN_FILENO, &n, sizeof(n)) != sizeof(n)) {
        std::cerr << "Ошибка при чтении размера массива" << std::endl;
        return 1;
    }

    std::vector<int> arr(n);

    if (read(STDIN_FILENO, arr.data(), n * sizeof(int)) != n * sizeof(int)) {
        std::cerr << "Ошибка при чтении элементов массива" << std::endl;
        return 1;
    }

    #ifdef BENCHMARK
    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();
    #endif /* BENCHMARK */

    oddEvenSort(arr);

    #ifdef BENCHMARK
    stop = std::chrono::system_clock::now();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << time << " us\n";
    #endif /* BENCHMARK */
    return 0;
}
