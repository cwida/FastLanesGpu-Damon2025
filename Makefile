# ==================
# Configure these vars
# ==================

# Replace with the Compute capability of your target GPU
COMPUTE_CAPABILITY := 61

# Replace with wherever you installed cuda
CUDA_LIBRARY_PATH := /usr/local/cuda-12.5

# NVCOMP Path
NVCOMP_INC := /usr/include/nvcomp_12/

# NVCOMP Path
NVCOMP_LIB := /usr/lib/x86_64-linux-gnu/nvcomp/12

# Unfortunately we need both clang (for ALP) and g++ (for nvCOMP)
#
# ALP requires clang
CLANG_PATH := /usr/bin/clang++-14
# g++ version has to be g++-12 or lower to compile nvCOMP
GCC_PATH := /usr/bin/g++-12

# ==================
# Some vars
# ==================

INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CUDA_OBJ_FLAGS := $(INC) $(LIB) 
OPTIMIZATION_LEVEL := -O3
CLANG_FLAGS := -std=c++17 -g $(WARNINGS) $(OPTIMIZATION_LEVEL)
WARNINGS := -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-padded -Wno-float-equal -Wno-global-constructors -Wno-exit-time-destructors 

#3033-D: inline variables are a C++17 feature
#3356-D: structured bindings are a C++17 feature
NVCC_IGNORE_ERR_NUMBERS := 3033,3356
CUDA_WARNING_FLAGS := -Wno-c++17-extensions
CUDA_FLAGS := --std=c++17 -ccbin $(CLANG_PATH) $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS) $(CUSTOM_FLAGS)
GCC_CUDA_FLAGS := --std=c++17 -ccbin $(GCC_PATH) $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS) $(CUSTOM_FLAGS)

ENGINE_HEADER_FILES := $(wildcard src/engine/*.cuh)
FLSGPU_HEADER_FILES := $(wildcard src/flsgpu/*.cuh)
NVCOMP_HEADER_FILES := $(wildcard src/flsgpu/*.cuh)
HEADER_FILES := src/alp/alp-bindings.cuh $(wildcard src/flsgpu/*.cuh) $(wildcard src/engine/*.cuh)

FLS_OBJ := $(patsubst src/fls/%.cpp, obj/fls-%.o, $(wildcard src/fls/*.cpp))
ALP_OBJ := $(patsubst src/alp/%.cpp, obj/alp-%.o, $(wildcard src/alp/*.cpp))
GENERATED_BINDINGS_OBJ := $(patsubst src/generated-bindings/%.cu, obj/generated-bindings-%.o, $(wildcard src/generated-bindings/*.cu))
NVCOMP_OBJ := $(patsubst src/nvcomp/%.cu, obj/nvcomp-%.o, $(wildcard src/nvcomp/*.cu))

SOURCE_FILES := obj/alp-bindings.o obj/enums.o $(FLS_OBJ) $(ALP_OBJ) 

# ==================
# OBJ Files
# ==================

obj/fls-%.o: src/fls/%.cpp
	$(CLANG_PATH) $^  -c -o $@ $(CLANG_FLAGS)

obj/alp-%.o: src/alp/%.cpp
	$(CLANG_PATH) $^  -c -o $@ $(CLANG_FLAGS)

obj/generated-bindings-%.o: src/generated-bindings/%.cu $(FLSGPU_HEADER_FILES) src/generated-bindings/multi-column-device-kernels.cuh src/engine/device-utils.cuh src/engine/kernels.cuh src/engine/multi-column-host-kernels.cuh
	nvcc $(word 1, $^) -c -o $@ $(CUDA_FLAGS) 

obj/enums.o: src/engine/enums.cu src/engine/enums.cuh
	nvcc $(word 1, $^) -c -o $@ $(CUDA_FLAGS) 

obj/alp-bindings.o: src/alp/alp-bindings.cu $(ALP_OBJ) 
	nvcc $(CUDA_FLAGS) -c -o $@ src/alp/alp-bindings.cu 

obj/nvcomp-%.o: src/nvcomp/%.cu $(NVCOMP_HEADER_FILES) 
	nvcc $(word 1, $^) -c -o $@ $(GCC_CUDA_FLAGS) -I$(NVCOMP_INC) -L$(NVCOMP_LIB) -lnvcomp 

# ==================
# Executables
# ==================

micro-benchmarks: src/micro-benchmarks.cu $(SOURCE_FILES) $(HEADER_FILES) $(GENERATED_BINDINGS_OBJ)
	nvcc $(CUDA_FLAGS) -g -o bin/$@ src/micro-benchmarks.cu $(SOURCE_FILES) $(GENERATED_BINDINGS_OBJ)

compressors-benchmarks: src/compressors-benchmarks.cu $(SOURCE_FILES) $(HEADER_FILES) $(NVCOMP_OBJ)
	nvcc $(GCC_CUDA_FLAGS) -g -I$(NVCOMP_INC) -L$(NVCOMP_LIB) -lnvcomp -o bin/$@ src/compressors-benchmarks.cu $(SOURCE_FILES) $(NVCOMP_OBJ)

generate: 
	./code-generators/generate-multicolumn-kernels.py
	./code-generators/generate-kernel-bindings.py

test-ffor: 
	./test-scripts/test-fls.sh

test-alp:
	./test-scripts/test-alp.sh

test-multi-column:
	./test-scripts/test-multi-column.sh

test-compressors:
	./test-scripts/test-compressors.sh

benchmark-all:
	sudo ./benchmark-scripts/run-benchmarks.py all benchmark-results/output/

benchmark-compressors:
	sudo ./benchmark-scripts/run-benchmarks.py compressors benchmark-results/output/

# ==================
# General
# ==================

setup: generate

test: test-ffor test-alp test-compressors test-multi-column 

all: micro-benchmarks compressors-benchmarks

clean:
	rm -f bin/*
	rm -f obj/*
