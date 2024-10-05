#include <iostream>
#include <cmath>
#include <iomanip>

const float EPS = 1e-7;

bool equalZero(float a) {
    return (a < EPS && a > -EPS);
}

void solve(float a, float b, float c) {
    if (equalZero(a) && equalZero(b)) {
        if (equalZero(c))
            std::cout << "any\n";
        else
            std::cout << "incorrect\n";
        return;
    }
    if (equalZero(a)) {
        std::cout << -c / b << "\n";
        return;
    }
    float d = b*b - 4*a*c;
    if (equalZero(d)) {
        std::cout << (-b) / (2*a) << "\n";
        return;
    }
    if (d > 0) {
        std::cout << (-b+std::sqrt(d)) / (2*a) << " " << (-b-std::sqrt(d)) / (2*a) << "\n";
        return;
    }
    std::cout << "imaginary\n";
}

int main() {
    float a, b, c;
    std::cin >> a >> b >> c;
    std::cout << std::setprecision(6) << std::fixed;
    solve(a, b, c);
}
