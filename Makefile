NVCC = nvcc
NVCCFLAGS = --std=c++11 -g -O2
CPP = g++
CPPFLAGS = --std=c++11 -g
BIN = lab1
OBJ = lab.cu

all: build

build:
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(BIN)

build_cpu:
	$(CPP) $(CPPFLAGS) lab_cpu.cpp -o lab1_cpu

clean:
	rm -rf *.o lab1_cpu $(BIN)
