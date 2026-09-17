# =============================================================================
# ==                    Makefile for the CUDA-Transport test suite         ==
# =============================================================================
# Invoked via `make test` from the repo root (which passes HDF5_INC/HDF5_LIB/
# CUDA_ARCH/HOST_COMPILER through), or directly as `make -C tests`.
#
# Each tests/test_*.cu is a small, self-contained binary with its own main()
# (see common.h for the tiny pass/fail framework) that exits nonzero if any
# of its checks fail. `make test` builds all of them and runs each in turn,
# failing overall if any one does.

HDF5_INC      ?= /usr/include/hdf5/serial
HDF5_LIB      ?= /usr/lib/x86_64-linux-gnu/hdf5/serial
CUDA_ARCH     ?= sm_80
HOST_COMPILER ?= g++

NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -arch=$(CUDA_ARCH) -ccbin $(HOST_COMPILER) -rdc=true
INCLUDES = -I../cuda_dynamics_lib/include -I../maps -I. -I$(HDF5_INC)
LDFLAGS = -L$(HDF5_LIB) -Xlinker -rpath -Xlinker $(HDF5_LIB) -lhdf5_cpp -lhdf5

HEADERS := $(wildcard ../cuda_dynamics_lib/include/*.h) \
           $(wildcard ../cuda_dynamics_lib/include/*.cuh) \
           $(wildcard ../cuda_dynamics_lib/include/solvers/*.cuh) \
           $(wildcard ../maps/*.h) \
           common.h

SRCS := $(wildcard test_*.cu)
BINS := $(patsubst %.cu,bin/%,$(SRCS))

.PHONY: all run clean
all: run

bin/%: %.cu $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

run: $(BINS)
	@fail=0; \
	for b in $(BINS); do \
		echo "=== $$b ==="; \
		./$$b || fail=1; \
		echo; \
	done; \
	if [ $$fail -eq 0 ]; then echo "=== ALL TEST BINARIES PASSED ==="; \
	else echo "=== AT LEAST ONE TEST BINARY FAILED ==="; fi; \
	exit $$fail

clean:
	rm -rf bin
