#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include "FileOperations.cuh"
#include "Constants.cuh"
#include "CPUCompressions.cuh" 
#include "GPUCompressionsFixed.cuh"
#include "GPUCompressionsRun.cuh"



void GPUFLCompressionProcess(uint8_t* input, int inputSize, char* outputFile)
{
    cudaSetDevice(0);
    uint8_t * Lengths;
    uint8_t * compressed;
    int compressedLength;
    GPUCompressionFL(input, inputSize, Lengths, compressed, compressedLength);
    // for(int i = 0; i < compressedLength; i++)
    // {
    //     printf("%d\n", compressed[i]);
    // }
    writeByteArrayToFileFL(outputFile, compressed, Lengths, compressedLength, inputSize);
    free(compressed);
    free(Lengths);
}

void GPUFLDecompressionProcess(char* inputFile, char* outputFile)
{
       cudaSetDevice(0);
       uint8_t* input, * output, * Lengths;
       int compressedSize, decompressedSize;
       readFileToByteArrayFL(inputFile, input, Lengths, compressedSize, decompressedSize);
       printf("read\n");
       GPUDecompressionFL(input, Lengths, output, compressedSize, decompressedSize);
    //    for(int i = 0; i < decompressedSize; i++)
    //    {
    //         printf("%d\n", output[i]);
    //    }
       writeByteArrayToFile(outputFile, output, decompressedSize);
}

void GPURLCompressionProcess(uint8_t* input, int inputSize, char* outputFile)
{
    cudaSetDevice(0);
    uint8_t* values;
    uint8_t* counts;
    int outputSize;

    GPUCompressionRL(input, inputSize, values, counts, outputSize);

    // for(int i = 0; i < outputSize; i++)
    // {
    //     printf("value: %d, count: %d\n", values[i], counts[i]);
    // }
    writeByteArrayToFileRL(outputFile, values, counts, outputSize, inputSize);
    free(counts);
    free(values);
}

void GPURLDecompressionProcess(char* inputFile, char* outputFile)
{
    cudaSetDevice(0);
    uint8_t* values, * output, * counts;
    int compressedSize, decompressedSize;
    readFileToByteArrayRL(inputFile, values, counts, compressedSize, decompressedSize);
    printf("read\n");
    GPUDecompressionRL(values, counts, compressedSize, output, decompressedSize);
    writeByteArrayToFile(outputFile, output, decompressedSize);
}

int main(int argc, char**argv)
{
   if(argc != 5)
   {
        printf("invalid number of arguments\n");
        return 0;
   }
   if(strcmp(argv[1], "c") == 0)
   {
        uint8_t* input;
        int inputSize = readFileToByteArray(argv[3], input);
        if(inputSize == -1)
        {
            exit(0);
        }
        // for(int i = 0; i < inputSize; i++)
        // {
        //     printf("%d\n", input[i]);
        // }
        //TODO: leak pamięci jeżeli argv[2] jest złe
        if(strcmp(argv[2], "fl") == 0)
        {
            GPUFLCompressionProcess(input, inputSize, argv[4]);
            free(input);
        }
        else if(strcmp(argv[2], "rl") == 0)
        {
            GPURLCompressionProcess(input, inputSize, argv[4]);
            free(input);
        }
        else
        {
            printf("unknown method\n");
        }

   }
   else if(strcmp(argv[1], "d") == 0)
   {
        if(strcmp(argv[2], "fl") == 0)
        {
             GPUFLDecompressionProcess(argv[3], argv[4]);
        }
        else if(strcmp(argv[2], "rl") == 0)
        {
            //TODO: naprawic
            GPURLDecompressionProcess(argv[3], argv[4]);
        }
        else
        {
            printf("unknown method\n");
        }
   }
   else
   {
        printf(argv[1]);
   }

}