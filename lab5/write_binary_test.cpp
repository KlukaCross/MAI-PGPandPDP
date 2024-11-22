#include <iostream>
#include <string>
#include <ctime>
using namespace std;

int main() {
    srand(time(0));

    int n;
    string filename;
    cin >> n;
    cin >> filename;

    int* data = (int*)malloc(sizeof(int) * n);
    FILE *f = fopen(filename.c_str(), "wb");
    fwrite(&n, sizeof(int), 1, f);
    for (int i=0; i < n; ++i) {
        int v = rand();
        fwrite(&v, sizeof(int), 1, f);
    }
    fclose(f);
}
