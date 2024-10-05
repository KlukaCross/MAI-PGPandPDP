#include <iostream>
#include <iomanip>
#include <vector>

void bubbleSort(std::vector<float>& v) {
    int n = v.size();
    bool isChange = true;
    while (isChange) {
        isChange = false;
        for (int i = 1; i < n; ++i) {
            if (v[i-1] > v[i]) {
                float tmp = v[i];
                v[i] = v[i-1];
                v[i-1] = tmp;
                isChange = true;
            }
        }
    }
}

int main() {
    int n;
    std::cin >> n;
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i)
        std::cin >> v[i];
    bubbleSort(v);
    std::cout << std::setprecision(6) << std::scientific;
    for (int i = 0; i < n; ++i)
        std::cout << v[i] << " ";
    std::cout << "\n";
}
