#include "GPUCompressionsRun.cuh"

__global__ void markBeginnings(uint8_t* d_input, int* d_beginnings, int* d_inputSize)
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

__global__ void divideTooLongSegments(int * d_beginnings_marks, int* d_beginingsIndexesFirst, int * d_lengths, int* d_outputSizeFirst)
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

void GPUCompressionRL(uint8_t* input, int inputSize, uint8_t*& values, uint8_t*& counts, int& outputSize)
{
    uint8_t* d_input = nullptr;
    int* d_beginingsIndexesFirst = nullptr;
    int* d_beginingsIndexes = nullptr;
    int* d_inputSize = nullptr;
    int* d_outputSizeFirst = nullptr;
    int* d_outputSize = nullptr;
    uint8_t* d_counts = nullptr;
    uint8_t* d_values = nullptr;
    thrust::device_vector<int> d_beginnings_marks(inputSize);
    thrust::device_vector<int> d_beginnings(inputSize);
    thrust::plus<int> binary_op;
    thrust::constant_iterator<int> ones(1);
    int blockSize = ceil(inputSize / 1024.0);
    int outputSizeFirst;
    thrust::device_vector<int> d_lengths;
    cudaError_t status;

    // Alokacja pamięci na GPU
    status = cudaMalloc(&d_input, sizeof(uint8_t) * inputSize);
    if (status != cudaSuccess) { printf("Error allocating d_input\n"); goto Error; }

    status = cudaMalloc(&d_inputSize, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_inputSize\n"); goto Error; }

    status = cudaMalloc(&d_outputSizeFirst, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_outputSizeFirst\n"); goto Error; }

    status = cudaMalloc(&d_outputSize, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_outputSize\n"); goto Error; }

    // Przekopiowanie 
    status = cudaMemcpy(d_input, input, sizeof(uint8_t) * inputSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying input to d_input\n"); goto Error; }

    status = cudaMemcpy(d_inputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying inputSize to d_inputSize\n"); goto Error; }

    // Zaznaczenie jedynkami w wektorze d_beginnings_marks początków kolejnych serii bajtów o tych samych wartościach
    // Bez uwzględnienia maksymalnej długości serii
    markBeginnings<<<blockSize, 1024>>>(d_input, thrust::raw_pointer_cast(d_beginnings_marks.data()), d_inputSize);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching markBeginnings kernel\n"); goto Error; }

    // Obliczamy długość outputu w przypadki gdy długości serii tych samych bajtów mogą być dowolnej wielkości
    // Potrzebne do podzielenia każdej z tych serii na odpowiednią ilość podserii
    outputSizeFirst = thrust::reduce(d_beginnings_marks.begin(), d_beginnings_marks.end(), 0, binary_op);
    status = cudaMemcpy(d_outputSizeFirst, &outputSizeFirst, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying outputSizeFirst to d_outputSizeFirst\n"); goto Error; }

    // w tablicy d_length dostajemy długości kolejnych serii tych samych bajtów
    thrust::inclusive_scan(d_beginnings_marks.begin(), d_beginnings_marks.end(), d_beginnings.begin(), binary_op);
    d_lengths.resize(outputSizeFirst);
    thrust::reduce_by_key(d_beginnings.begin(), d_beginnings.end(), ones, thrust::make_discard_iterator(), d_lengths.begin(), thrust::equal_to<int>());

    // obliczenie indeksów początków serii bez ograniczenia na długość serii (potrzebne aby następnie podzielić za długie serie)
    status = cudaMalloc(&d_beginingsIndexesFirst, sizeof(int) * outputSizeFirst);
    if (status != cudaSuccess) { printf("Error allocating d_beginingsIndexesFirst\n"); goto Error; }
    setBeginningIndexes<<<blockSize, 1024>>>(d_beginingsIndexesFirst, thrust::raw_pointer_cast(d_beginnings.data()), d_inputSize);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching setBeginningIndexes kernel\n"); goto Error; }

    // w tablicy d_lengths po tej transformacji znajduje się informacja na ile serii należy podzielić każdą z serii
    thrust::transform(d_lengths.begin(), d_lengths.end(), d_lengths.begin(), ceil_divide_by_256());

    // dołożenie do tablicy d_beginnings_marks jedynek oznaczających początki serii powstałych z podzielenia zbyt długich serii
    blockSize = ceil(outputSizeFirst / 1024.0);
    divideTooLongSegments<<<blockSize, 1024>>>(thrust::raw_pointer_cast(d_beginnings_marks.data()), d_beginingsIndexesFirst, thrust::raw_pointer_cast(d_lengths.data()), d_outputSizeFirst);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching divideTooLongSegments kernel\n"); goto Error; }

    // obliczenie długości tablic z długościami i wartościami
    outputSize = thrust::reduce(d_beginnings_marks.begin(), d_beginnings_marks.end(), 0, binary_op);
    status = cudaMemcpy(d_outputSize, &outputSize, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying outputSize to d_outputSize\n"); goto Error; }

    thrust::inclusive_scan(d_beginnings_marks.begin(), d_beginnings_marks.end(), d_beginnings.begin(), binary_op);

    // inicjalizacja tablicy z indeksami początków serii
    status = cudaMalloc(&d_beginingsIndexes, sizeof(int) * (outputSize + 1));
    if (status != cudaSuccess) { printf("Error allocating d_beginingsIndexes\n"); goto Error; }

    status = cudaMemcpy(d_beginingsIndexes + outputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying inputSize to d_beginingsIndexes\n"); goto Error; }

    //obliczenie indeksów początków serii
    blockSize = ceil(inputSize / 1024.0);
    setBeginningIndexes<<<blockSize, 1024>>>(d_beginingsIndexes, thrust::raw_pointer_cast(d_beginnings.data()), d_inputSize);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching setBeginningIndexes kernel\n"); goto Error; }

    // inicjalizacja tablic reprezentujących zkompresowany plik
    counts = (uint8_t*)malloc(sizeof(uint8_t) * outputSize);
    values = (uint8_t*)malloc(sizeof(uint8_t) * outputSize);

    status = cudaMalloc(&d_counts, sizeof(uint8_t) * outputSize);
    if (status != cudaSuccess) { printf("Error allocating d_counts\n"); goto Error; }

    status = cudaMalloc(&d_values, sizeof(uint8_t) * outputSize);
    if (status != cudaSuccess) { printf("Error allocating d_values\n"); goto Error; }

    // kompresja pliku
    blockSize = ceil(outputSize / 1024.0);
    fillCompressedArrays<<<blockSize, 1024>>>(d_counts, d_values, d_beginingsIndexes, d_input, d_outputSize);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching fillCompressedArrays kernel\n"); goto Error; }

    // przekopiowanie zkompresowamegi pliku z GPU na CPU
    status = cudaMemcpy(counts, d_counts, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying d_counts to counts\n"); goto Error; }

    status = cudaMemcpy(values, d_values, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying d_values to values\n"); goto Error; }

Error:
    cudaFree(d_counts);
    cudaFree(d_values);
    cudaFree(d_input);
    cudaFree(d_beginingsIndexesFirst);
    cudaFree(d_beginingsIndexes);
    cudaFree(d_inputSize);
    cudaFree(d_outputSizeFirst);
    cudaFree(d_outputSize);
}

void GPUDecompressionRL(uint8_t* values, uint8_t* counts, int size, uint8_t*& output, int inputSize)
{
    thrust::device_vector<int> d_counts(counts, counts + size);
    thrust::device_vector<int> d_beginnings(size);
    thrust::plus<int> binary_op;
    int blockSize = ceil(size / 1024.0);
    thrust::inclusive_scan(d_counts.begin(), d_counts.end() - 1, d_beginnings.begin() + 1, binary_op);
    int outputSize = thrust::reduce(d_counts.begin(), d_counts.end(), 0, binary_op);
    uint8_t* d_output = nullptr;
    uint8_t* d_values = nullptr;
    int* d_size = nullptr;
    cudaError_t status;

    // Alokacja potrzebnych tablic na GPU
    status = cudaMalloc(&d_output, sizeof(uint8_t) * outputSize);
    if (status != cudaSuccess) { printf("Error allocating d_output\n"); goto Error; }

    status = cudaMalloc(&d_values, sizeof(uint8_t) * size);
    if (status != cudaSuccess) { printf("Error allocating d_values\n"); goto Error; }

    status = cudaMalloc(&d_size, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_size\n"); goto Error; }

    // Inicjalizacja danych na GPU
    status = cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying size to d_size\n"); goto Error; }

    status = cudaMemcpy(d_values, values, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying values to d_values\n"); goto Error; }

    // Dekompresja
    fillDecompressedOutput<<<blockSize, 1024>>>(d_values, thrust::raw_pointer_cast(d_counts.data()), d_output, thrust::raw_pointer_cast(d_beginnings.data()), d_size);
    status = cudaGetLastError();
    if (status != cudaSuccess) { printf("Error launching fillDecompressedOutput kernel\n"); goto Error; }

    output = (uint8_t*)malloc(sizeof(uint8_t) * outputSize);
    if (output == nullptr) {
        printf("Error allocating host memory for output\n");
        goto Error;
    }

    // Przekopiownie zkompresowanego pliku z GPU na CPU

    status = cudaMemcpy(output, d_output, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying d_output to output\n"); goto Error; }

Error:
    cudaFree(d_output);
    cudaFree(d_values);
    cudaFree(d_size);
}
