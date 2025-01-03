#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
 

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int k, id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    for (k = id; k < n; k += blockDim.x*gridDim.x)
      {
        c[k] = a[k] + b[k];
      }
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 50000;
 
    // Size of vectors
    int blockSize = 1024;
    int gridSize = 10;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    cudaEvent_t start,stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    // Number of threads in each thread block
    //blockSize = 1024;
 
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, blockSize);
    
    cudaEventRecord(start,0);
    
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
 
    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
 
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf ("Time for the kernel: %f ms\n", elapsedTime);
    
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
