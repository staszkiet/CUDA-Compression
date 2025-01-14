#include "GPUCompressionsFixed.cuh"

__global__ void findCompressionLengths(uint8_t* d_buff, uint8_t* d_lengths, int* d_segmentsCount, int* d_inputSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_segmentsCount)
    {
        int max = 0;
        int offset = index*SEGMENT_SIZE;
        //znalezienie elementu maksymalnego w segmencie
        for(int i = 0; i < SEGMENT_SIZE && i + offset < *d_inputSize; i++)
        {
            if(d_buff[i + offset] > max)
            {
                max = d_buff[i + offset];
            }
        }

        //obliczenie ilości bitów potrzebnych do zakodowania danych w segmencie
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

__global__ void fillCompressedArray(uint8_t* d_compressed, uint8_t* d_buff, int* d_beginnigs, uint8_t* d_lengths, int* d_inputSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < *d_inputSize)
    {
        int positionInSegment = index%SEGMENT_SIZE;
        int segmentIdx = index/SEGMENT_SIZE;

        //indeks bitów w tablicy wynikowej mówiące o tym od którego do którego bitu jest miejsce dla skompresowanej
        //wartości elementu którym zajmuje się wątek. Zkompresowany element leży w przedziale <startPosition, endPosition).
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

            // każda operacja bitowa zapisana jest do noewj zmiennej pomocnieczej aby na pewno uciąć jedynki które wypadły poza zakres 
            if(startPosition/8 == endPosition/8)
            {
                uint8_t pom = d_compressed[byteIndex];
                int leftOffset = startPosition - byteIndex*8;
                uint8_t pom1 =  (pom << leftOffset);
                d_output[index] = pom1 >> (8 - d_lengths[segmentIdx]);
            }
            else
            {
                uint8_t right = d_compressed[byteIndex + 1];
                uint8_t left = d_compressed[byteIndex];
                uint8_t rightOffset = (byteIndex + 2)*8 - endPosition;
                uint8_t pom = (left << (startPosition - byteIndex*8));
                uint8_t pom2 = pom >> (8 - d_lengths[segmentIdx]);
                uint8_t pom3 = (right >> rightOffset);
                d_output[index] = pom3 | pom2;
            }
        }
    
}

void GPUDecompressionFL(uint8_t* compressed, uint8_t * Lengths, uint8_t*& output, int compressedLength, int decompressedSize)
{
    int segmentsCount = ceil(decompressedSize/static_cast<double>(SEGMENT_SIZE));
    int blockSize = ceil(decompressedSize/1024.0);

    uint8_t* d_compressed, *d_output;
    int * d_decompressedSize;
    thrust::device_vector<uint8_t>d_Lengths(Lengths, Lengths + segmentsCount);
    thrust::device_vector<int> d_beginnings(segmentsCount);
    cudaError_t status;

    //alokacja potrzebnych zasobów

    output = (uint8_t*)malloc(sizeof(uint8_t)*decompressedSize);
    if(output == nullptr){printf("Error allocating output\n"); goto Error;}

    status = cudaMalloc(&d_compressed, sizeof(uint8_t)*compressedLength);
    if (status != cudaSuccess) { printf("Error allocating d_compressed\n"); goto Error; }

    status = cudaMalloc(&d_output, sizeof(uint8_t)*decompressedSize);
    if (status != cudaSuccess) { printf("Error allocating d_output\n"); goto Error; }

    status = cudaMalloc(&d_decompressedSize, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_decompressedSize\n"); goto Error; }

    //inicjalizacja danymi tabic na GPU

    status = cudaMemcpy(d_compressed, compressed, sizeof(uint8_t)*compressedLength, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_compressed\n"); goto Error; }

    status = cudaMemcpy(d_decompressedSize, &decompressedSize, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_decompressedSize\n"); goto Error; }

    status = cudaMemset(d_output, 0, sizeof(uint8_t)*decompressedSize);
    if (status != cudaSuccess) { printf("Error memset on d_output\n"); goto Error; }

    // stworzenie tablicy pozwalającej obliczyć na którym bicie w kolejności rozpoczyna się dany segment
    // (trzeba przemnożyć wartość razy rozmiar segmentu). trzeba d_Lengths najpierw zamienić na inty żeby
    // zapobiec problemom z overflow

    thrust::transform(d_Lengths.begin(), d_Lengths.end(), d_beginnings.begin(), thrust::identity<int>());
    thrust::exclusive_scan(d_beginnings.begin(), d_beginnings.end(), d_beginnings.begin());

    // kernel wykonujący dekompresję FL i zapisujący zdekompresowany plik do tablicy d_output

    fillDecompressedArray<<<blockSize, 1024>>>(d_compressed, thrust::raw_pointer_cast(d_Lengths.data()), thrust::raw_pointer_cast(d_beginnings.data()), d_output, d_decompressedSize);

    // przepisanie wyniku z GPU na CPU

    status = cudaMemcpy(output, d_output, sizeof(uint8_t)*decompressedSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying from d_output\n"); goto Error; }

Error:
    cudaFree(d_compressed);
    cudaFree(d_output);
    cudaFree(d_decompressedSize);
}

void GPUCompressionFL(uint8_t * input, int inputSize, uint8_t*& Lengths, uint8_t*& compressed, int& compressedLength)
{
    compressedLength = 0;

    int segmentsCount = ceil(inputSize/static_cast<double>(SEGMENT_SIZE));
    int blockSize = ceil(segmentsCount/1024.0);
    int lastSegmentLength;

    uint8_t * d_input;
    uint8_t * d_compressed;
    int * d_segmentsCount;
    int * d_inputSize;
    thrust::device_vector<int> d_beginnings(segmentsCount);
    cudaError_t status;
    thrust::device_vector<uint8_t> d_Lengths(segmentsCount);

    //alokacja pamięci

    Lengths = (uint8_t*)malloc(sizeof(uint8_t)*segmentsCount);
    if(Lengths == nullptr){printf("Error allocating Lengths\n"); goto Error;}

    status = cudaMalloc(&d_input, sizeof(uint8_t)*inputSize);
    if (status != cudaSuccess) { printf("Error allocating d_input\n"); goto Error; }

    status = cudaMalloc(&d_segmentsCount, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_segmentsCount\n"); goto Error; }

    status = cudaMalloc(&d_inputSize, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_inputSize\n"); goto Error; }

    //kopiowanie danych z CPU na GPU

    status = cudaMemcpy(d_input, input, sizeof(uint8_t)*inputSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_input\n"); goto Error; }

    status = cudaMemcpy(d_segmentsCount, &segmentsCount, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_segmentsCount\n"); goto Error; }

    status = cudaMemcpy(d_inputSize, &inputSize, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_inputSize\n"); goto Error; }

    //kernel oblicza i zapiusje w wektorze d_Lengths na ilu bitach będą kodowane kolejne segmenty

    findCompressionLengths<<<blockSize, 1024>>>(d_input, thrust::raw_pointer_cast(d_Lengths.data()), d_segmentsCount, d_inputSize);

    // stworzenie tablicy pozwalającej obliczyć na którym bicie w kolejności rozpoczyna się dany segment
    // (trzeba przemnożyć wartość razy rozmiar segmentu). trzeba d_Lengths najpierw zamienić na inty żeby
    // zapobiec problemom z overflow

    thrust::copy(d_Lengths.begin(), d_Lengths.end(), Lengths);
    thrust::transform(d_Lengths.begin(), d_Lengths.end(), d_beginnings.begin(), thrust::identity<int>());
    thrust::exclusive_scan(d_beginnings.begin(), d_beginnings.end(), d_beginnings.begin());

    //obliczenie ile bajtów trzeba zaalokować dla tablicy z skompresowanym outputem

    lastSegmentLength = (inputSize - SEGMENT_SIZE*(segmentsCount - 1))*Lengths[segmentsCount - 1];
    compressedLength = lastSegmentLength + d_beginnings[segmentsCount - 1]*SEGMENT_SIZE;
    compressedLength = ceil(compressedLength/8.0);

    //alokacja i wyczyszczenie tablicy z outputem

    status = cudaMalloc(&d_compressed, sizeof(uint8_t)*compressedLength);
    if (status != cudaSuccess) { printf("Error allocating d_compressed\n"); goto Error; }

    status = cudaMemset(d_compressed, 0, sizeof(uint8_t)*compressedLength);
    if (status != cudaSuccess) { printf("Error memset on d_compressed\n"); goto Error; }

    compressed = (uint8_t*)malloc(sizeof(uint8_t)*compressedLength);
    if(compressed == nullptr){printf("Error allocating compressed\n"); goto Error;}

    blockSize = ceil(inputSize/1024.0);

    //kernel wykonujący kompresję FL i wpisujący zkompresowane dane do tablicy d_compressed

    fillCompressedArray<<<blockSize, 1024>>>(d_compressed, d_input, thrust::raw_pointer_cast(d_beginnings.data()), thrust::raw_pointer_cast(d_Lengths.data()), d_inputSize);

    //przepisanie wyniku z GPU na CPU

    status = cudaMemcpy(compressed, d_compressed, sizeof(uint8_t)*compressedLength, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying from d_compressed\n"); goto Error; }

Error:
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_segmentsCount);
    cudaFree(d_inputSize);
}
