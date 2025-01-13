CC=nvcc
C_FLAGS_NODE3=--std=c++20
SRC=main.cu
TARGET=CompressionAlg

all:
	${CC} ${C_FLAGS_NODE3} -o ${TARGET} ${SRC}

clean:
	rm -f ${TARGET}

.PHONY: clean