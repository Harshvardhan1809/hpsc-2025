#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void  init(int *bucket){
  bucket[threadIdx.x] = 0;
}

__global__ void buck(int *bucket, int *key){
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

int main() {
  int n = 50;
  int range = 5;

  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n"); 

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));

  init<<<1,range>>>(bucket);
  cudaDeviceSynchronize();

  buck<<<1, n>>>(bucket, key);
  cudaDeviceSynchronize();

  // not straight-forward to parallelize; for less iterations serial is okay
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}