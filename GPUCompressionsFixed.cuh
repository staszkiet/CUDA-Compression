#ifndef GPUFL_H
#define GPUFL_H

#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include "Constants.cuh"
#include <thrust/device_vector.h>

__global__ void findCompressionLengths(uint8_t* d_buff, uint8_t* d_lengths, int* d_segmentsCount, int* d_inputSize);

__device__ uint8_t atomicOrChar(uint8_t* address, uint8_t val);

__global__ void fillCompressedArray(uint8_t* d_compressed, uint8_t* d_buff, int* d_beginnigs, uint8_t* d_lengths, int* d_compressedSize);

__global__ void fillDecompressedArray(uint8_t* d_compressed, uint8_t* d_lengths, int* d_beginnings, uint8_t* d_output, int* d_decompressedSize);

void GPUDecompressionFL(uint8_t* compressed, uint8_t * Lengths, uint8_t*& output, int compressedLength, int decompressedSize);

void GPUCompressionFL(uint8_t * input, int inputSize, uint8_t*& Lengths, uint8_t*& compressed, int& compressedLength);

#endif