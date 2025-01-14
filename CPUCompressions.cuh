#ifndef CPU_H
#define CPU_H

#include <bits/stdc++.h>
#include "Constants.cuh"

void CPUCompressionFL(uint8_t * input, uint8_t *& output, int*& Lengths);

void CPUDecompressionFL(uint8_t* compressed, uint8_t* output, int * Lengths);

void CPUCompressionRL(uint8_t * buff, uint8_t *& valuesTab, uint8_t *& countsTab, int& size);

void CPUDecompressionRL(uint8_t* values, uint8_t* counts, int size, uint8_t *& output);

#endif