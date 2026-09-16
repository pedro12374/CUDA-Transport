# =============================================================================
# ==          Makefile for the CUDA Dynamical Systems Library              ==
# =============================================================================
#
# This is a generic Makefile for public distribution.
# To use it, configure the variables in the "USER CONFIGURATION" section below
# (or override them on the command line, e.g. `make HDF5_INC=/opt/hdf5/include`).
#
# =============================================================================
# ## USER CONFIGURATION ##
#
# Instructions:
#   1. Set HDF5_INC / HDF5_LIB to your HDF5 installation's include/lib dirs.
#      Defaults below match the Debian/Ubuntu "serial" HDF5 package layout.
#   2. Set CUDA_ARCH to the compute capability of your GPU.
#      - Find your GPU's code here: https://developer.nvidia.com/cuda-gpus
#      - Examples: sm_80 (A30/A100), sm_86 (RTX 3080), sm_89 (RTX 4090)
#   3. Set HOST_COMPILER to your preferred C++ host compiler.
#
# =============================================================================

HDF5_INC      ?= /usr/include/hdf5/serial
HDF5_LIB      ?= /usr/lib/x86_64-linux-gnu/hdf5/serial
CUDA_ARCH     ?= sm_80
HOST_COMPILER ?= g++

# =============================================================================
# ## COMPILER SETUP (AUTOMATIC) ##
# - No need to edit below this line for basic configuration -
# =============================================================================
# Compiler
NVCC = nvcc

# Compiler flags are built from the user-configured variables
CXXFLAGS = -std=c++17 -O3 -arch=$(CUDA_ARCH) -ccbin $(HOST_COMPILER) -rdc=true

# Include paths
INCLUDES = -I./cuda_dynamics_lib/include \
           -I./maps \
           -I$(HDF5_INC)

# Library paths and libraries to link (HDF5 C++ API + core C library)
LDFLAGS = -L$(HDF5_LIB) -Xlinker -rpath -Xlinker $(HDF5_LIB) -lhdf5_cpp -lhdf5

# =============================================================================
# ## PROJECT STRUCTURE (AUTOMATIC) ##
# =============================================================================

# All library/map headers, so touching any of them triggers a rebuild.
HEADERS := $(wildcard cuda_dynamics_lib/include/*.h) \
           $(wildcard cuda_dynamics_lib/include/*.cuh) \
           $(wildcard cuda_dynamics_lib/include/solvers/*.cuh) \
           $(wildcard maps/*.h)

# =============================================================================
# ## BUILD RULES ##
# =============================================================================
#
# Each dynamical system gets ONE generic driver binary (run_<system>),
# config-file-driven (see configs/*.cfg) rather than one hand-written .cu
# per calculation. Adding a new system: write its header under maps/ (the
# functor + Traits + a load_params_for<> specialization -- see maps/horton.h
# for a worked ODE example, maps/standard_map.h for a discrete-map one),
# write a 4-line entry point .cu like run_horton.cu, and add a rule below.

# Define all executables you want to build
all: run_horton run_standard_map run_henon_heiles

run_horton: run_horton.cu $(HEADERS)
	@echo "==> Building Horton generic runner"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

run_standard_map: run_standard_map.cu $(HEADERS)
	@echo "==> Building Standard Map generic runner"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

run_henon_heiles: run_henon_heiles.cu $(HEADERS)
	@echo "==> Building Henon-Heiles generic runner"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# TUTORIAL.md's worked example (a driven, damped pendulum). Not part of
# `all` -- it's a teaching example, not one of the library's maintained
# systems -- but `make run_pendulum` builds it directly.
run_pendulum: run_pendulum.cu $(HEADERS)
	@echo "==> Building Pendulum generic runner"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Add other rules for other executables here...

# Rule to clean up all compiled files
clean:
	@echo "==> Cleaning up build files..."
	rm -f run_horton run_standard_map run_henon_heiles run_pendulum
	$(MAKE) -C tests clean

# Builds and runs the test suite (see tests/README.md). Each test is a
# small, fast (well under a minute total), self-contained .cu that exits
# nonzero on failure.
test:
	$(MAKE) -C tests \
		HDF5_INC="$(HDF5_INC)" HDF5_LIB="$(HDF5_LIB)" \
		CUDA_ARCH="$(CUDA_ARCH)" HOST_COMPILER="$(HOST_COMPILER)"

# Phony targets are not files
.PHONY: all clean test
