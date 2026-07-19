# Usage:
#   make                 # build every example into build/
#   make ARCH=sm_86      # pick an arch (RTX 30 = sm_86, 40 = sm_89, 50 = sm_120)
#   make run             # build and run everything, each example self-checks
#   make clean

ARCH ?= sm_86
NVCC ?= nvcc
NVCCFLAGS := -O3 -std=c++17 -arch $(ARCH) -Icommon

CU_SRCS := $(shell find . -name '*.cu' | sort)
BINS := $(patsubst ./%.cu,build/%,$(CU_SRCS))

# extra libs where needed
build/08_gemm_opt/sgemm_vs_cublas: EXTRA_LIBS := -lcublas

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
