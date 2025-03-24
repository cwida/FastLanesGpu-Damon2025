INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CUDA_OBJ_FLAGS = $(INC) $(LIB) 
OPTIMIZATION_LEVEL = -O3
WARNINGS = -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-padded -Wno-float-equal -Wno-global-constructors -Wno-exit-time-destructors 
CLANG_FLAGS = -std=c++17 -g $(WARNINGS) $(OPTIMIZATION_LEVEL)
GCC_FLAGS = -std=c++17 -g $(WARNINGS) $(OPTIMIZATION_LEVEL)

#3033-D: inline variables are a C++17 feature
#3356-D: structured bindings are a C++17 feature
NVCC_IGNORE_ERR_NUMBERS=3033,3356
CUDA_WARNING_FLAGS=-Wno-c++17-extensions
COMPUTE_CAPABILITY = 70
#CUDA_FLAGS = -ccbin /usr/bin/clang++-14 $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS)
#GCC required
NVCOMP_INC ?= /usr/include/nvcomp_12/
NVCOMP_LIB ?= /usr/lib/x86_64-linux-gnu/nvcomp/12
CUDA_FLAGS = -ccbin /usr/bin/g++-12 -I$(NVCOMP_INC) -L$(NVCOMP_LIB) -lnvcomp -std=c++17 $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS)

GPU_CUH := $(wildcard src/gpu/*.cuh)
GPU_OBJ := $(patsubst src/gpu/%.cu, obj/gpu-%.o, $(wildcard src/gpu/*.cu))
FLS_OBJ := $(patsubst src/fls/%.cpp, obj/fls-%.o, $(wildcard src/fls/*.cpp))
ALP_OBJ := $(patsubst src/alp/%.cpp, obj/alp-%.o, $(wildcard src/alp/*.cpp))

# OBJ Files
obj/fls-%.o: src/fls/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/alp-%.o: src/alp/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/gpu-%.o: src/gpu/%.cu  $(GPU_CUH)
	nvcc $(CUDA_FLAGS) -c -o $@ $(word 1, $^)

obj/benchmark-compressors.o: src/benchmark-compressors.cu src/benchmark-compressors.cuh
	nvcc $(CUDA_FLAGS) -c -o $@ $(word 1, $^)

obj/nvcomp-compressors.o: src/nvcomp-compressors.cu src/nvcomp-compressors.cuh
	nvcc $(CUDA_FLAGS) -c -o $@ $(word 1, $^)

# Executables
HEADER_FILES=$(wildcard src/**.h) $(wildcard src/gpu/*.hpp)
OBJ_FILES=$(FLS_OBJ) $(ALP_OBJ) $(GPU_OBJ) obj/nvcomp-compressors.o obj/benchmark-compressors.o

micro-benchmarks: $(OBJ_FILES) $(HEADER_FILES)
	nvcc src/micro-benchmarks.cu $(OBJ_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CUDA_FLAGS) $(CUDA_OBJ_FLAGS) 

multi-column-benchmarks: $(OBJ_FILES) $(HEADER_FILES)
	nvcc src/multi-column-benchmarks.cu $(OBJ_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CUDA_FLAGS) $(CUDA_OBJ_FLAGS) 

benchmark-single-compressor: $(OBJ_FILES) $(HEADER_FILES)
	nvcc src/benchmark-single-compressor.cu $(OBJ_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CUDA_FLAGS) $(CUDA_OBJ_FLAGS) 

benchmark-all-compressors: $(OBJ_FILES) $(HEADER_FILES)
	nvcc src/benchmark-all-compressors.cu $(OBJ_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CUDA_FLAGS) $(CUDA_OBJ_FLAGS) 

all: micro-benchmarks multi-column-benchmarks benchmark-single-compressor benchmark-all-compressors

clean:
	rm -f bin/*
	rm -f obj/*
