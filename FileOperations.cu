#include "FileOperations.cuh"


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

int writeByteArrayToFileFL(char* filename, uint8_t* data, uint8_t* Lengths, int dataSize, int inputSize)
{ 

    int segmentsCount = ceil(inputSize/static_cast<double>(SEGMENT_SIZE));
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
    // for(int i = 0; i < segmentsCount; i++)
    // {
    //     printf("Length: %d\n", Lengths[i]);
    // }
    bytesWritten = fwrite(Lengths, sizeof(uint8_t), segmentsCount, file);
    if (bytesWritten != segmentsCount) {
        perror("Failed to write Lengths to file");
        fclose(file);
        return -1;
    }

    fclose(file);

    return 0;
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
    // for(int i = 0; i < compressedSize; i++)
    // {
    //     printf("value: %d count: %d\n", values[i], counts[i]);
    // }

    return compressedSize;
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
    int segmentsCount = ceil(decompressedSize/static_cast<double>(SEGMENT_SIZE));

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
    // for(int i = 0; i < segmentsCount; i++)
    // {
    //     printf("Lengths: %d\n", Lengths[i]);
    // }

    return compressedSize;
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