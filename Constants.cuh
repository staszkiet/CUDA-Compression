#define INPUT_SIZE 1000
#define SEGMENT_SIZE 8
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <bits/stdc++.h>
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}  
#endif