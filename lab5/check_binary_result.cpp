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
    for (int i=1; i < n; ++i) {
        if (data[i-1] > data[i]) {
            std::cout << "FAILED!\n" << "indexes: " << i-1 << " " << i << "\ndata: " << data[i-1] << ">" << data[i] << "\n";
            return 1;
        }
    }
    std::cout << "SUCCESS!\n";
    return 0;
}
