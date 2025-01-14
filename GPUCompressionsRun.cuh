#ifndef GPURUN_H
#define GPURUN_H
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include "Constants.cuh"

__global__ void markBeginnings(uint8_t* d_input, uint8_t* d_beginnings, int* d_inputSize);

__global__ void setBeginningIndexes(int* d_beginingsIndexes, int* d_beginnings, int* d_inputSize);

__global__ void fillCompressedArrays(uint8_t* d_counts, uint8_t* d_values, int* d_beginingsIndexes, uint8_t* d_input, int *d_outputSize);

__global__ void divideTooLongSegments(uint8_t * d_beginnings_marks, int* d_beginingsIndexesFirst, int * d_lengths, int* d_outputSizeFirst);


struct ceil_divide_by_256
{
    __host__ __device__
    int operator()(int x) const {
        return std::ceil(x / 256.0); // Ceiling of x / 256
    }
};

__global__ void fillDecompressedOutput(uint8_t* d_values, int* d_counts, uint8_t* d_output, int* d_beginnings, int* d_size);

void GPUCompressionRL(uint8_t* input, int inputSize, uint8_t*& values, uint8_t*& counts, int & outputSize);

void GPUDecompressionRL(uint8_t* values, uint8_t* counts, int size, uint8_t*& output, int inputSize);

#endif