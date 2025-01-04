# NVCC = /usr/local/cuda/bin/nvcc

NVCC = nvcc
FLAGS_NVCC = -O2 -std=c++11 -Xcompiler -Wno-error -lm

BIN = cw

SOURCES = src/main.cu
HEADERS = $(wildcard src/**/*.cuh)

MAX_FRAMES = 8192

all: $(SOURCES) $(HEADERS)
	$(NVCC) $(FLAGS_NVCC) $(SOURCES) -o $(BIN)

$(BIN): $(SOURCES) $(HEADERS)
	$(NVCC) $(FLAGS_NVCC) $(SOURCES) -o $(BIN)

run_gpu: $(BIN)
	bash scripts/start.sh $(BIN) $(MAX_FRAMES) "--gpu"

run_cpu: $(BIN)
	bash scripts/start.sh $(BIN) $(MAX_FRAMES) "--cpu"

clean:
	rm -rf *.o $(BIN)
