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
           $(wildcard cuda_dynamics_lib/include/solvers/*.cuh) \
           $(wildcard maps/*.h)

# =============================================================================
# ## BUILD RULES ##
# =============================================================================

# Define all executables you want to build
all: horton_msd horton_escape

# Rule to build the Horton escape time simulator
horton_escape: main_horton_escape.cu $(HEADERS)
	@echo "==> Building Horton Escape simulator"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Rule to build the Horton MSD simulator
horton_msd: main_horton_msd.cu $(HEADERS)
	@echo "==> Building Horton MSD simulator"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Add other rules for other executables here...

# Rule to clean up all compiled files
clean:
	@echo "==> Cleaning up build files..."
	rm -f horton_escape horton_msd

# Phony targets are not files
.PHONY: all clean
