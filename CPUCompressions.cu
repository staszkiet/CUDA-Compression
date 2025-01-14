#include "CPUCompressions.cuh"

void CPUCompressionFL(uint8_t * input, uint8_t *& output, int*& Lengths)
{
    int segments_count = ceil(INPUT_SIZE/static_cast<double>(SEGMENT_SIZE));
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
    int segmentsCount = ceil(INPUT_SIZE/static_cast<double>(SEGMENT_SIZE));
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

void CPUCompressionRL(uint8_t * buff, uint8_t *& valuesTab, uint8_t *& countsTab, int& size)
{
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