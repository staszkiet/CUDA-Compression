# Compiler
CC = nvcc

# Compiler flags
C_FLAGS_NODE3 = --std=c++20

# Source files
SRC = main.cu FileOperations.cu CPUCompressions.cu GPUCompressionsFixed.cu GPUCompressionsRun.cu

# Target executable name
TARGET = CompressionAlg

# Build rule
all:
	${CC} ${C_FLAGS_NODE3} -o ${TARGET} ${SRC}

# Clean rule
clean:
	rm -f ${TARGET}

.PHONY: clean