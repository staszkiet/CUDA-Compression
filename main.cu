#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/iterator/constant_iterator.h>
#include <bits/stdc++.h>
#include <thrust/iterator/discard_iterator.h>

#define INPUT_SIZE 1000
#define SEGMENT_SIZE 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}   


void CPUCompressionFL(uint8_t * input, uint8_t *& output, int*& Lengths)
{
    int segments_count = ceil(INPUT_SIZE/static_cast<float>(SEGMENT_SIZE));
    int * Maxes;
    int compressedLength = 0;
    int lastSegmentLength;
    Lengths = (int*)malloc(sizeof(int)*segments_count);
    Maxes = (int*)malloc(sizeof(int)*segments_count);
    memset(Maxes, 0, sizeof(int)*segments_count);
    //printf("segments count: %d\n", segments_count);

    
    for(int i = 0; i <= segments_count; i++)
    {
        //znalezienie max dla segmentu
        for(int j = 0; j < SEGMENT_SIZE && i*SEGMENT_SIZE + j < INPUT_SIZE; j++)
        {
            if(input[i*SEGMENT_SIZE + j] > Maxes[i])
            {
                Maxes[i] = input[i*SEGMENT_SIZE + j];
            }
        }
        int l = 2;
        //znalezienie długości kompresji dla danego max
        for(int j = 1; j <= 8; j++)
        {
            if(Maxes[i] < l)
            {
                Lengths[i] = j;
                break;
            }
            l = l * 2;
        }
    }

    for(int i = 0; i < INPUT_SIZE/SEGMENT_SIZE; i++)
    {
        compressedLength += SEGMENT_SIZE*Lengths[i];
    }
    lastSegmentLength = (INPUT_SIZE - SEGMENT_SIZE*(segments_count - 1))*Lengths[segments_count - 1];
    compressedLength += lastSegmentLength;
    compressedLength = ceil(compressedLength/8.0);
    printf("compressedLength: %d\n", compressedLength);

    output = (uint8_t*)malloc(compressedLength*sizeof(uint8_t));

    //kompresja
    int startPosition = 0;
    int endPosition;
    for(int i = 0; i <= segments_count; i++)
    {
        for(int j = 0; j < SEGMENT_SIZE && i*SEGMENT_SIZE + j < INPUT_SIZE; j++)
        {
            endPosition = startPosition + Lengths[i];
            if(endPosition/8 == startPosition/8)
            {
                
                int byteIndex = endPosition/8;
                int offset = (byteIndex+1)*8 - endPosition;
                output[byteIndex] = output[byteIndex] | input[i*SEGMENT_SIZE + j] << offset;
            }
            else
            {
                int byteIndex = startPosition/8;
                int rightOffset = (byteIndex + 2)*8 - endPosition;
                int leftOffset = 8 - rightOffset;
                output[byteIndex] = output[byteIndex] | input[i*SEGMENT_SIZE + j] >> leftOffset;
                output[byteIndex + 1] = output[byteIndex + 1] | input[i*SEGMENT_SIZE + j] << rightOffset;
            }
            startPosition = endPosition;
        }
    }
    for(int i = 0; i < compressedLength; i++)
    {
        printf("%d\n", output[i]);
    }
    free(Maxes);
}

void CPUDecompressionFL(uint8_t* compressed, uint8_t* output, int * Lengths)
{
    int segmentsCount = ceil(INPUT_SIZE/static_cast<float>(SEGMENT_SIZE));
    int startPosition = 0;
    int endPosition;
    for(int i = 0; i <= segmentsCount; i++)
    {
        for(int j = 0; j < SEGMENT_SIZE && i*SEGMENT_SIZE + j < INPUT_SIZE; j++)
        {
            endPosition = startPosition + Lengths[i];
            int byteIndex = startPosition/8;
            if(startPosition/8 == endPosition/8)
            {
                uint8_t pom = compressed[byteIndex];
                int leftOffset = startPosition - byteIndex*8;
               // printf("pom: %d index: %d start: %d, byteINdex8 :%d \n", pom, i*SEGMENT_SIZE + j, startPosition, byteIndex);
                uint8_t pom1 =  (pom << leftOffset);
                output[i*SEGMENT_SIZE + j] = pom1 >> (8 - Lengths[i]);
            }
            else
            {
                uint8_t right = compressed[byteIndex + 1];
                uint8_t left = compressed[byteIndex];
                uint8_t rightOffset = (byteIndex + 2)*8 - endPosition;
               // printf("index: %d, right: %d, left: %d start: %d, end: %d, %d\n", i*SEGMENT_SIZE + j, right, left, startPosition, endPosition, (left << (startPosition - byteIndex*8)) );
                uint8_t pom = (left << (startPosition - byteIndex*8));
                uint8_t pom2 = pom >> (8 - Lengths[i]);
                uint8_t pom3 = (right >> rightOffset);
                output[i*SEGMENT_SIZE + j] = pom3 | pom2;
            }
            startPosition = endPosition;
        }
    }
}

void CPUFLCompressionDecompressionTest()
{
    uint8_t buff[INPUT_SIZE];
    int* Lengths = nullptr;
    uint8_t * buffCompressed = nullptr;
    uint8_t buffDecompressed[INPUT_SIZE];
    //inicializacja danych do debugowania
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        buff[i] = i;
    }
    
    CPUCompressionFL(buff, buffCompressed, Lengths);

    CPUDecompressionFL(buffCompressed, buffDecompressed, Lengths);
    
    free(Lengths);
}

__global__ void findCompressionLengths(uint8_t* d_buff, uint8_t* d_lengths, int* d_segmentsCount)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_segmentsCount)
    {
        int max = 0;
        int offset = index*SEGMENT_SIZE;
        for(int i = 0; i < SEGMENT_SIZE && i + offset < INPUT_SIZE; i++)
        {
            if(d_buff[i + offset] > max)
            {
                max = d_buff[i + offset];
            }
        }
        int l = 2;
        for(int j = 1; j <= 8; j++)
        {
            if(max < l)
            {
                d_lengths[index] = j;
                break;
            }
            l = l * 2;
        }
    }

}

__device__ uint8_t atomicOrChar(uint8_t* address, uint8_t val)
{
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};     
    unsigned int sel = selectors[(size_t)address & 3];              
    unsigned int old, assumed, new_;

    old = *base_address;                                           

    do {
        assumed = old;

        uint8_t extracted_byte = (uint8_t)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440);
        uint8_t result = extracted_byte | val;

        new_ = __byte_perm(old, result, sel);

        if (new_ == old)                                           
            break;

        old = atomicCAS(base_address, assumed, new_);

    } while (assumed != old);    

    return (uint8_t)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440);   
}

__global__ void fillCompressedArray(uint8_t* d_compressed, uint8_t* d_buff, int* d_beginnigs, uint8_t* d_lengths, int* d_compressedSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_compressedSize)
    {
        int positionInSegment = index%SEGMENT_SIZE;
        int segmentIdx = index/SEGMENT_SIZE;
        int startPosition = d_beginnigs[segmentIdx]*SEGMENT_SIZE + d_lengths[segmentIdx]*positionInSegment;
        int endPosition = startPosition + d_lengths[segmentIdx];

            
        if(endPosition/8 == startPosition/8)
        {
            int byteIndex = endPosition/8;
            int offset = (byteIndex+1)*8 - endPosition;
            atomicOrChar(d_compressed + byteIndex, d_buff[index] << offset);
        }
        else
        {
            int byteIndex = startPosition/8;
            int rightOffset = (byteIndex + 2)*8 - endPosition;
            int leftOffset = 8 - rightOffset;
            atomicOrChar(d_compressed+byteIndex, d_buff[index] >> leftOffset);
            atomicOrChar(d_compressed + byteIndex + 1, d_buff[index] << rightOffset);
        }
    }
       
}

__global__ void fillDecompressedArray(uint8_t* d_compressed, uint8_t* d_lengths, int* d_beginnings, uint8_t* d_output, int* d_decompressedSize)
{
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if(index < *d_decompressedSize)
        {
            int positionInSegment = index%SEGMENT_SIZE;
            int segmentIdx = index/SEGMENT_SIZE;
            int startPosition = d_beginnings[segmentIdx]*SEGMENT_SIZE + d_lengths[segmentIdx]*positionInSegment;
            int endPosition = startPosition + d_lengths[segmentIdx];
            int byteIndex = startPosition/8;

            if(startPosition/8 == endPosition/8)
            {
                uint8_t pom = d_compressed[byteIndex];
                int leftOffset = startPosition - byteIndex*8;
                // printf("pom: %d index: %d start: %d, byteINdex8 :%d \n", pom, i*SEGMENT_SIZE + j, startPosition, byteIndex);
                uint8_t pom1 =  (pom << leftOffset);
                d_output[index] = pom1 >> (8 - d_lengths[segmentIdx]);
            }
            else
            {
                uint8_t right = d_compressed[byteIndex + 1];
                uint8_t left = d_compressed[byteIndex];
                uint8_t rightOffset = (byteIndex + 2)*8 - endPosition;
                // printf("index: %d, right: %d, left: %d start: %d, end: %d, %d\n", i*SEGMENT_SIZE + j, right, left, startPosition, endPosition, (left << (startPosition - byteIndex*8)) );
                uint8_t pom = (left << (startPosition - byteIndex*8));
                uint8_t pom2 = pom >> (8 - d_lengths[segmentIdx]);
                uint8_t pom3 = (right >> rightOffset);
                d_output[index] = pom3 | pom2;
            }
        }
    
}

void GPUDecompressionFL(uint8_t* compressed, uint8_t * Lengths, uint8_t*& output, int compressedLength, int decompressedSize)
{
    output = (uint8_t*)malloc(sizeof(uint8_t)*decompressedSize);
    int segmentsCount = ceil(decompressedSize/static_cast<float>(SEGMENT_SIZE));

    uint8_t* d_compressed, *d_output;
    int * d_decompressedSize;
    gpuErrchk(cudaMalloc(&d_compressed, sizeof(uint8_t)*compressedLength));
    gpuErrchk(cudaMalloc(&d_output, sizeof(uint8_t)*decompressedSize));
    gpuErrchk(cudaMalloc(&d_decompressedSize, sizeof(int)));
    gpuErrchk( cudaMemcpy(d_compressed, compressed, sizeof(uint8_t)*compressedLength, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_decompressedSize, &decompressedSize, sizeof(int), cudaMemcpyHostToDevice));
    printf("after memcpy\n");
    thrust::device_vector<uint8_t>d_Lengths(Lengths, Lengths + segmentsCount);
    thrust::device_vector<int> d_beginnings(segmentsCount);
    thrust::transform(d_Lengths.begin(), d_Lengths.end(), d_beginnings.begin(), thrust::identity<int>());

    printf("after cpy\n");
    thrust::exclusive_scan(d_beginnings.begin(), d_beginnings.end(), d_beginnings.begin());

    printf("before kernel\n");
    int blockSize = ceil(decompressedSize/1024.0);
    fillDecompressedArray<<<blockSize, 1024>>>(d_compressed, thrust::raw_pointer_cast(d_Lengths.data()), thrust::raw_pointer_cast(d_beginnings.data()), d_output, d_decompressedSize);
    printf("after kernell\n");
    gpuErrchk( cudaMemcpy(output, d_output, sizeof(uint8_t)*decompressedSize, cudaMemcpyDeviceToHost));
    cudaFree(d_compressed);
    cudaFree(d_output);

}

void GPUCompressionFL(uint8_t * input, int inputSize, uint8_t*& Lengths, uint8_t*& compressed, int& compressedLength)
{
    compressedLength = 0;
    int segmentsCount = ceil(inputSize/static_cast<float>(SEGMENT_SIZE));

    uint8_t * d_input;
    uint8_t * d_compressed;
    int * d_segmentsCount;
    int * d_inputSize;
    thrust::device_vector<uint8_t> d_Lengths(segmentsCount);
    Lengths = (uint8_t*)malloc(sizeof(uint8_t)*segmentsCount);
    gpuErrchk( cudaMalloc(&d_input, sizeof(uint8_t)*inputSize));
    gpuErrchk( cudaMalloc(&d_segmentsCount, sizeof(int)));
    gpuErrchk( cudaMalloc(&d_inputSize, sizeof(int)));
    gpuErrchk( cudaMemcpy(d_input, input, sizeof(uint8_t)*inputSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_segmentsCount, &segmentsCount, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_inputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = ceil(segmentsCount/1024.0);
    findCompressionLengths<<<blockSize, 1024>>>(d_input, thrust::raw_pointer_cast(d_Lengths.data()), d_segmentsCount);

    thrust::copy(d_Lengths.begin(), d_Lengths.end(), Lengths);

    //tablica początków 
    thrust::device_vector<int> d_beginnings(segmentsCount);
    thrust::transform(d_Lengths.begin(), d_Lengths.end(), d_beginnings.begin(), thrust::identity<int>());
    thrust::exclusive_scan(d_beginnings.begin(), d_beginnings.end(), d_beginnings.begin());

    for(int i = 0; i < segmentsCount; i++)
    {
        printf("length %d: %d\n", i, Lengths[i]);
    }
    

    //obliczenie długosci outputu
    int lastSegmentLength = (inputSize - SEGMENT_SIZE*(segmentsCount - 1))*Lengths[segmentsCount - 1];
    compressedLength = lastSegmentLength + d_beginnings[segmentsCount - 1]*SEGMENT_SIZE;
    compressedLength = ceil(compressedLength/8.0);
    gpuErrchk( cudaMalloc(&d_compressed, sizeof(uint8_t)*compressedLength));
    compressed = (uint8_t*)malloc(sizeof(uint8_t)*compressedLength);

    printf("compressedLength: %d\n", compressedLength);
    blockSize = ceil(inputSize/1024.0);
    fillCompressedArray<<<blockSize, 1024>>>(d_compressed, d_input, thrust::raw_pointer_cast(d_beginnings.data()), thrust::raw_pointer_cast(d_Lengths.data()), d_inputSize);
    gpuErrchk( cudaMemcpy(compressed, d_compressed, sizeof(uint8_t)*compressedLength, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_segmentsCount);
    cudaFree(d_inputSize);
}


void CPUCompressionRL(uint8_t * buff, uint8_t *& valuesTab, uint8_t *& countsTab, int& size)
{
    srand(0);
    std::vector<uint8_t> values;
    std::vector<uint8_t> counts;
    //inicializacja danych do debugowania

    int count = 0;
    uint8_t actual = buff[0];
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        if(buff[i] == actual && count < 255)
        {
            count++;
        }
        else
        {
            values.push_back(actual);
            counts.push_back(count);
            count = 1;
            actual = buff[i];
        }
    }
    values.push_back(actual);
    counts.push_back(count);
    valuesTab = (uint8_t*)malloc(sizeof(uint8_t)*values.size());
    countsTab = (uint8_t*)malloc(sizeof(uint8_t)*counts.size());
    size = values.size();
    for(int i = 0; i < size; i++)
    {
        valuesTab[i] = values[i];
        countsTab[i] = counts[i];
    }
}

void CPUDecompressionRL(uint8_t* values, uint8_t* counts, int size, uint8_t *& output)
{
    std::vector<uint8_t> v;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < counts[i]; j++)
        {
            v.push_back(values[i]);
        }
    }
    output = (uint8_t*)malloc(sizeof(uint8_t)*v.size());
    std::copy(v.begin(), v.end(), output);
}

void CPURLCompressionDecompressionTest()
{
    uint8_t buff[INPUT_SIZE];
    uint8_t * values;
    uint8_t * counts;
    uint8_t * decompression;
    int size;
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        buff[i] = rand() % 2;
        printf("%d", buff[i]);
    }
    printf("\n");

    CPUCompressionRL(buff, values, counts, size);
    CPUDecompressionRL(values, counts, size, decompression);
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        printf("%d", decompression[i]);
    }
    printf("\n");
    free(values);
    free(counts);
    free(decompression);
}

__global__ void markBeginnings(uint8_t* d_input, uint8_t* d_beginnings, int* d_inputSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_inputSize)
    {
        if(index == 0 || d_input[index] != d_input[index-1])
        {
            d_beginnings[index] = 1;
        }
    }
}

__global__ void setBeginningIndexes(int* d_beginingsIndexes, int* d_beginnings, int* d_inputSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_inputSize)
    {
        if(index != 0 && d_beginnings[index] != d_beginnings[index - 1])
        {
            d_beginingsIndexes[d_beginnings[index - 1]] = index;
        }
    }
}

__global__ void fillCompressedArrays(uint8_t* d_counts, uint8_t* d_values, int* d_beginingsIndexes, uint8_t* d_input, int *d_outputSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if(index - 1 < *d_outputSize)
    {
        d_counts[index - 1] = d_beginingsIndexes[index] - d_beginingsIndexes[index - 1];
        d_values[index - 1] = d_input[d_beginingsIndexes[index - 1]];
    }
}

__global__ void divideTooLongSegments(uint8_t * d_beginnings_marks, int* d_beginingsIndexesFirst, int * d_lengths, int* d_outputSizeFirst)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_outputSizeFirst)
    {
        for(int i = 1; i < d_lengths[index]; i++)
        {
            d_beginnings_marks[d_beginingsIndexesFirst[index] + 255*i] = 1;
        }
    }
}

struct ceil_divide_by_256
{
    __host__ __device__
    int operator()(int x) const {
        return std::ceil(x / 256.0); // Ceiling of x / 256
    }
};

void GPUCompressionRL(uint8_t* input, int inputSize, uint8_t*& values, uint8_t*& counts, int & outputSize)
{
    uint8_t* d_input;
    int* d_beginingsIndexesFirst, *d_beginingsIndexes;
    int* d_inputSize;
    int * d_outputSizeFirst;
    int * d_outputSize;
    uint8_t* d_counts;
    uint8_t* d_values;
    thrust::device_vector<uint8_t> d_beginnings_marks(inputSize);
    thrust::device_vector<int> d_beginnings(inputSize);
    int* beginningIndexes;

    gpuErrchk(cudaMalloc(&d_input, sizeof(uint8_t)*inputSize));
    gpuErrchk(cudaMalloc(&d_inputSize, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_outputSizeFirst, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_outputSize, sizeof(int)));
    gpuErrchk( cudaMemcpy(d_input, input, sizeof(uint8_t)*inputSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_inputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice));

    //zaznaczenie początków serii tam gdzie zmiena się wartość
    int blockSize = ceil(inputSize/1024.0);
    markBeginnings<<<blockSize, 1024>>>(d_input, thrust::raw_pointer_cast(d_beginnings_marks.data()), d_inputSize);
    thrust::plus<int> binary_op;

    //ilość serii dla dowolnej długości pojedynczej serii
    int outputSizeFirst = thrust::reduce(d_beginnings_marks.begin(), d_beginnings_marks.end(), 0, binary_op);
    gpuErrchk( cudaMemcpy(d_outputSizeFirst, &outputSizeFirst, sizeof(int), cudaMemcpyHostToDevice));


    thrust::inclusive_scan(d_beginnings_marks.begin(), d_beginnings_marks.end(), d_beginnings.begin(), binary_op);
    thrust::constant_iterator<int> ones(1);

    //długości kolejnych serii
    thrust::device_vector<int> d_lengths(outputSizeFirst);
    thrust::reduce_by_key(d_beginnings.begin(), d_beginnings.end(),  ones, thrust::make_discard_iterator(), d_lengths.begin(), thrust::equal_to<int>());

    gpuErrchk(cudaMalloc(&d_beginingsIndexesFirst, sizeof(int)*outputSizeFirst));

    setBeginningIndexes<<<blockSize, 1024>>>(d_beginingsIndexesFirst, thrust::raw_pointer_cast(d_beginnings.data()), d_inputSize);
    thrust::transform(d_lengths.begin(), d_lengths.end(), d_lengths.begin(), ceil_divide_by_256());
    blockSize = ceil(outputSizeFirst/1024.0);
    divideTooLongSegments<<<blockSize, 1024>>>(thrust::raw_pointer_cast(d_beginnings_marks.data()), d_beginingsIndexesFirst, thrust::raw_pointer_cast(d_lengths.data()), d_outputSizeFirst);
    outputSize = thrust::reduce(d_beginnings_marks.begin(), d_beginnings_marks.end(), 0, binary_op);
    gpuErrchk( cudaMemcpy(d_outputSize, &outputSize, sizeof(int), cudaMemcpyHostToDevice));
    printf("outputSize: %d\n", outputSize);
    thrust::inclusive_scan(d_beginnings_marks.begin(), d_beginnings_marks.end(), d_beginnings.begin(), binary_op);
    gpuErrchk(cudaMalloc(&d_beginingsIndexes, sizeof(int)*(outputSize + 1)));
    gpuErrchk( cudaMemcpy(d_beginingsIndexes + outputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice));
    blockSize = ceil(inputSize/1024.0);
    setBeginningIndexes<<<blockSize, 1024>>>(d_beginingsIndexes, thrust::raw_pointer_cast(d_beginnings.data()), d_inputSize);


    counts = (uint8_t*)malloc(sizeof(uint8_t)*outputSize);
    values = (uint8_t*)malloc(sizeof(uint8_t)*outputSize);
    beginningIndexes = (int*)malloc(sizeof(int)*(outputSize + 1));
    gpuErrchk(cudaMalloc(&d_counts, sizeof(uint8_t)*outputSize));
    gpuErrchk(cudaMalloc(&d_values, sizeof(uint8_t)*outputSize));
    blockSize = ceil(outputSize/1024.0);
    fillCompressedArrays<<<blockSize, 1024>>>(d_counts, d_values, d_beginingsIndexes, d_input, d_outputSize);
    gpuErrchk(cudaMemcpy(counts, d_counts, sizeof(uint8_t)*outputSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(values, d_values, sizeof(uint8_t)*outputSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(beginningIndexes, d_beginingsIndexes, sizeof(int)*(outputSize + 1), cudaMemcpyDeviceToHost));
    free(beginningIndexes);
    cudaFree(d_counts);
    cudaFree(d_values);
    cudaFree(d_input);
    cudaFree(d_beginingsIndexesFirst);
    cudaFree(d_inputSize);
    cudaFree(d_outputSizeFirst);
}

__global__ void fillDecompressedOutput(uint8_t* d_values, int* d_counts, uint8_t* d_output, int* d_beginnings, int* d_size)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_size)
    {
        for(int i = d_beginnings[index]; i < d_beginnings[index] + d_counts[index]; i++)
        {
            d_output[i] = d_values[index];
        }
    }
}

void GPUDecompressionRL(uint8_t* values, uint8_t* counts, int size, uint8_t*& output, int inputSize) //TODO: czy inputSize potrzebne??
{
    thrust::device_vector<int> d_counts(counts, counts + size);
    thrust::device_vector<int> d_beginnings(size);
    thrust::plus<int> binary_op;
    int outputSize = thrust::reduce(d_counts.begin(), d_counts.end(), 0, binary_op);
    thrust::inclusive_scan(d_counts.begin(), d_counts.end() - 1, d_beginnings.begin() + 1, binary_op);
    int * d_size;

    uint8_t* d_output, * d_values;

    cudaMalloc(&d_output, sizeof(uint8_t)*outputSize);
    cudaMalloc(&d_values, sizeof(uint8_t)*size);
    cudaMalloc(&d_size, sizeof(int));
    gpuErrchk(cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_values, values, sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
    int blockSize = ceil(size/1024.0);
    fillDecompressedOutput<<<blockSize, 1024>>>(d_values, thrust::raw_pointer_cast(d_counts.data()), d_output,thrust::raw_pointer_cast(d_beginnings.data()), d_size);
    output = (uint8_t*)malloc(sizeof(uint8_t)*outputSize);
    gpuErrchk(cudaMemcpy(output, d_output, sizeof(uint8_t)*outputSize, cudaMemcpyDeviceToHost)); 
    printf("\n");
    cudaFree(d_output);
    cudaFree(d_values);
    cudaFree(d_size);

}

void GPURLCompressionDecompressionTest()
{
    uint8_t input[INPUT_SIZE];
    uint8_t* values;
    uint8_t* counts;
    uint8_t* decompressed;
    int outputSize;

    for(int i = 0; i < 100; i++)
    {
        input[i] = 2;
        printf("%d", input[i]);
    }
    for(int i = 100; i < INPUT_SIZE; i++)
    {
        input[i] = 10;
        printf("%d", input[i]);
    }
    printf("\n");
    GPUCompressionRL(input, INPUT_SIZE, values, counts, outputSize);
    for(int i = 0; i < outputSize; i++)
    {
         printf("value: %d count: %d\n", values[i], counts[i]);
    }
    GPUDecompressionRL(values, counts, outputSize, decompressed, INPUT_SIZE);
    free(counts);
    free(values);
    free(decompressed);
}

int readFileToByteArray(char* filename, uint8_t*& input)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    rewind(file);
    if(size < 0)
    {
        perror("Couldnt find the size");
        fclose(file);
        return -1;
    }

    input = (uint8_t *)malloc(size);
    size_t bytesRead = fread(input, sizeof(uint8_t), size, file);
    if(bytesRead != size)
    {
        perror("Not all contents were read");
        fclose(file);
        free(input);
        return -1;
    }

    return size;
}

int readFileToByteArrayFL(char* filename, uint8_t*& input, uint8_t*& Lengths, int& compressedSize, int& decompressedSize)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }
    size_t bytesRead = fread(&compressedSize, sizeof(int), 1, file);
    if(bytesRead != 1)
    {
        perror("Not all contents were read");
        fclose(file);
        return -1;
    }
    printf("compressed size: %d\n", compressedSize);
    bytesRead = fread(&decompressedSize, sizeof(int), 1, file);
    if(bytesRead != 1)
    {
        perror("Not all contents were read");
        fclose(file);
        return -1;
    }
    printf("decompressed size: %d\n", decompressedSize);
    int segmentsCount = ceil(decompressedSize/static_cast<float>(SEGMENT_SIZE));

    input = (uint8_t *)malloc(compressedSize*sizeof(uint8_t));
    Lengths = (uint8_t *)malloc(segmentsCount*sizeof(uint8_t));
    bytesRead = fread(input, sizeof(uint8_t), compressedSize, file);
    if(bytesRead != compressedSize)
    {
        perror("Not all contents were read");
        fclose(file);
        free(Lengths);
        free(input);
        return -1;
    }
    bytesRead = fread(Lengths, sizeof(uint8_t), segmentsCount, file);
    if(bytesRead != segmentsCount)
    {
        perror("Not all contents were read");
        free(Lengths);
        free(input);
        fclose(file);
        return -1;
    }
    for(int i = 0; i < segmentsCount; i++)
    {
        printf("Lengths: %d\n", Lengths[i]);
    }

    return compressedSize;
}

int readFileToByteArrayRL(char* filename, uint8_t*& values, uint8_t*& counts, int& compressedSize, int& decompressedSize)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }
    size_t bytesRead = fread(&compressedSize, sizeof(int), 1, file);
    if(bytesRead != 1)
    {
        perror("Not all contents were read");
        fclose(file);
        return -1;
    }
    printf("compressed size: %d\n", compressedSize);
    bytesRead = fread(&decompressedSize, sizeof(int), 1, file);
    if(bytesRead != 1)
    {
        perror("Not all contents were read");
        fclose(file);
        return -1;
    }
    printf("decompressed size: %d\n", decompressedSize);

    values = (uint8_t *)malloc(compressedSize*sizeof(uint8_t));
    counts = (uint8_t *)malloc(compressedSize*sizeof(uint8_t));
    bytesRead = fread(values, sizeof(uint8_t), compressedSize, file);
    if(bytesRead != compressedSize)
    {
        perror("Not all contents were read");
        fclose(file);
        free(values);
        free(counts);
        return -1;
    }
    bytesRead = fread(counts, sizeof(uint8_t), compressedSize, file);
    if(bytesRead != compressedSize)
    {
        perror("Not all contents were read");
        free(counts);
        free(values);
        fclose(file);
        return -1;
    }
    for(int i = 0; i < compressedSize; i++)
    {
        printf("value: %d count: %d\n", values[i], counts[i]);
    }

    return compressedSize;
}

int writeByteArrayToFileFL(char* filename, uint8_t* data, uint8_t* Lengths, int dataSize, int inputSize)
{ 

    int segmentsCount = ceil(inputSize/static_cast<float>(SEGMENT_SIZE));
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Failed to open file for writing");
        return -1;
    }

    size_t bytesWritten = fwrite(&dataSize, sizeof(int), 1, file);
    if (bytesWritten != 1) {
        perror("Failed to write complete data to file");
        fclose(file);
        return -1;
    }
    bytesWritten = fwrite(&inputSize, sizeof(int), 1, file);
    if (bytesWritten != 1) {
        perror("Failed to write complete data to file");
        fclose(file);
        return -1;
    }
    bytesWritten = fwrite(data, sizeof(uint8_t), dataSize, file);
    if (bytesWritten != dataSize) {
        perror("Failed to write data to file");
        fclose(file);
        return -1;
    }
    for(int i = 0; i < segmentsCount; i++)
    {
        printf("Length: %d\n", Lengths[i]);
    }
    bytesWritten = fwrite(Lengths, sizeof(uint8_t), segmentsCount, file);
    if (bytesWritten != segmentsCount) {
        perror("Failed to write Lengths to file");
        fclose(file);
        return -1;
    }

    fclose(file);

    return 0;
}

int writeByteArrayToFile(char* filename, uint8_t* data, int dataSize)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Failed to open file for writing");
        return -1;
    }

    size_t bytesWritten = fwrite(data, sizeof(uint8_t), dataSize, file);
    if (bytesWritten != dataSize) {
        perror("Failed to write data to file");
        fclose(file);
        return -1;
    }

    fclose(file);

    return 0;
}

int writeByteArrayToFileRL(char* filename, uint8_t* values, uint8_t* counts, int dataSize, int inputSize)
{ 
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Failed to open file for writing");
        return -1;
    }

    size_t bytesWritten = fwrite(&dataSize, sizeof(int), 1, file);
    if (bytesWritten != 1) {
        perror("Failed to write complete data to file");
        fclose(file);
        return -1;
    }
    bytesWritten = fwrite(&inputSize, sizeof(int), 1, file);
    if (bytesWritten != 1) {
        perror("Failed to write complete data to file");
        fclose(file);
        return -1;
    }
    bytesWritten = fwrite(values, sizeof(uint8_t), dataSize, file);
    if (bytesWritten != dataSize) {
        perror("Failed to write data to file");
        fclose(file);
        return -1;
    }

    bytesWritten = fwrite(counts, sizeof(uint8_t), dataSize, file);
    if (bytesWritten != dataSize) {
        perror("Failed to write data to file");
        fclose(file);
        return -1;
    }

    fclose(file);

    return 0;
}

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
       for(int i = 0; i < decompressedSize; i++)
       {
            printf("%d\n", output[i]);
       }
       writeByteArrayToFile(outputFile, output, decompressedSize);
}

void GPURLCompressionProcess(uint8_t* input, int inputSize, char* outputFile)
{
    cudaSetDevice(0);
    uint8_t* values;
    uint8_t* counts;
    int outputSize;

    GPUCompressionRL(input, inputSize, values, counts, outputSize);

    for(int i = 0; i < outputSize; i++)
    {
        printf("value: %d, count: %d\n", values[i], counts[i]);
    }
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
        for(int i = 0; i < inputSize; i++)
        {
            printf("%d\n", input[i]);
        }
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