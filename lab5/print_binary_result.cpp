#include <iostream>
#include <string>

int main() {
    int n;
    std::string filename;
    std::cin >> n;
    std::cin >> filename;
    int* data = (int*)malloc(sizeof(int) * n);
    FILE *f = fopen(filename.c_str(), "rb");
    fread(data, sizeof(int), n, f);
    fclose(f);
    for (int i=0; i < n; ++i) {
        std::cout << data[i] << " ";
    }
}
