#ifndef FILEOP_H
#define FILEOP_H

#include <bits/stdc++.h>
#include "Constants.cuh"

int writeByteArrayToFileRL(char* filename, uint8_t* values, uint8_t* counts, int dataSize, int inputSize);

int writeByteArrayToFile(char* filename, uint8_t* data, int dataSize);

int writeByteArrayToFileFL(char* filename, uint8_t* data, uint8_t* Lengths, int dataSize, int inputSize);

int readFileToByteArrayRL(char* filename, uint8_t*& values, uint8_t*& counts, int& compressedSize, int& decompressedSize);

int readFileToByteArrayFL(char* filename, uint8_t*& input, uint8_t*& Lengths, int& compressedSize, int& decompressedSize);

int readFileToByteArray(char* filename, uint8_t*& input);

#endif