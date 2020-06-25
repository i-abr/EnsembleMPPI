
#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // allocate unified memory
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // run the kernel on 1M elements
    add<<<1, 256>>>(N, x, y);

    cudaDeviceSynchronize();

    float max_error = 0.0f;
    for (int i = 0; i < N; i++)
        max_error = fmax(max_error, fabs(y[i]-3.0f));

    std::cout << "max error " << max_error <<std::endl;

    // free cuda memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
