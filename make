IDIR=./
COMPILER=nvcc

# Libraries required for the project
LIBRARIES += -lcuda -lcudart -lcufft

# Compiler flags
COMPILER_FLAGS=-I/u/local/cuda/11.8/include -I/u/local/cuda/11.8/lib64 \
	-I./aquila ${LIBRARIES} --std c++11 \
	-g

# Name of the output executable
OUTPUT=fft

# Source and object files
SRC=src/fft.cu
OBJ=bin/fft.o

# Build rule
build: bin/$(OUTPUT)

bin/$(OUTPUT): $(SRC)
	$(COMPILER) $(COMPILER_FLAGS) $< -o $@

# Clean rule
clean:
	rm -f bin/*

# Run rule
run:
	./bin/$(OUTPUT) $(ARGS)
