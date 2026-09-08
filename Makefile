# Usage:
#   make                 # build every example into build/
#   make ARCH=sm_86      # pick an arch (RTX 30 = sm_86, 40 = sm_89, 50 = sm_120)
#   make run             # build and run everything, each example self-checks
#   make clean

ARCH ?= sm_86
NVCC ?= nvcc
NVCCFLAGS := -O3 -std=c++17 -arch $(ARCH) -Icommon

CU_SRCS := $(shell find . -name '*.cu' | sort)
# 15_torch_extension builds through torch's cpp_extension instead
# (python3 15_torch_extension/build_and_benchmark.py), not through make
CU_SRCS := $(filter-out ./15_torch_extension/%,$(CU_SRCS))

# extra libs where needed
build/08_gemm_opt/sgemm_vs_cublas: EXTRA_LIBS := -lcublas
build/08_gemm_opt/sgemm_cutlass: EXTRA_LIBS := -lcublas

# CUTLASS (header-only) is optional: sgemm_cutlass only builds when the
# library is found, so a fresh clone without it still does `make` cleanly.
# Get it with:  git clone --depth 1 -b v4.8.0 https://github.com/NVIDIA/cutlass ~/cutlass
CUTLASS_DIR ?= $(HOME)/cutlass
ifneq ($(wildcard $(CUTLASS_DIR)/include/cutlass/cutlass.h),)
# diag-suppress: cutlass's own conv3d header trips #20013 on CUDA 13, not my code
build/08_gemm_opt/sgemm_cutlass: NVCCFLAGS += -I$(CUTLASS_DIR)/include -diag-suppress 20013,20015
else
CU_SRCS := $(filter-out ./08_gemm_opt/sgemm_cutlass.cu,$(CU_SRCS))
endif

# BINS has to be computed *after* the CUTLASS filter above, or a fresh
# clone without cutlass would still try to build sgemm_cutlass (and fail)
BINS := $(patsubst ./%.cu,build/%,$(CU_SRCS))

.PHONY: all run clean

all: $(BINS)

build/%: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(EXTRA_LIBS)

run: all
	@for bin in $(BINS); do \
		echo "===== $$bin ====="; \
		./$$bin || exit 1; \
		echo; \
	done

clean:
	rm -rf build
