#include <iostream>
#include <cstdlib>

__global__
void cuda_add(const int *ar, const int *br, int *cr) {
  const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int a = ar[idx];
  const int b = br[idx];
  cr[idx] = a + b;
}

__host__
int main(void) {

  const unsigned threads = 1<<16;
  const unsigned size = threads*sizeof(int);

  // Initialize host arrays
  int *a_host = new int[threads];
  int *b_host = new int[threads];
  int *c_host = new int[threads];
  for (unsigned i=0; i<threads; i++) {
    a_host[i] = (std::rand()%10);
    b_host[i] = (std::rand()%10);
  }

  // Initialize device arrays
  int *a_dev = NULL;
  int *b_dev = NULL;
  int *c_dev = NULL;
  cudaMalloc((void**)&a_dev,size);
  cudaMalloc((void**)&b_dev,size);
  cudaMalloc((void**)&c_dev,size);
  
  // Transfer memory
  cudaMemcpy((void*)a_dev,(void*)a_host,size,cudaMemcpyHostToDevice);
  cudaMemcpy((void*)b_dev,(void*)b_host,size,cudaMemcpyHostToDevice);

  // Setup and launch kernel
  const unsigned threads_per_block = 512;
  const unsigned blocks_per_grid = threads / threads_per_block;
  cuda_add<<<threads_per_block,blocks_per_grid>>>(a_dev,b_dev,c_dev);

  // Copy back result and print it
  cudaMemcpy((void*)c_host,(void*)c_dev,size,cudaMemcpyDeviceToHost);
  for (size_t i=0; i<threads; i++) std::cout << c_host[i] << " ";
  std::cout << std::endl;

  // Clean up
  delete a_host;
  delete b_host;
  delete c_host;
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);

  return 0;
}