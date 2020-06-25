#include <iostream>
#include <cmath>
#include <armadillo>

int main(){
    const int size = 256;
    double sinTable[size];

    #pragma omp parallel for
    for (int n=0; n <size; ++n)
        sinTable[n] = std::sin(2 * M_PI * n / size);

    for (int n = 0; n < size; ++n)
        std::cout << sinTable[n] << std::endl;
}
