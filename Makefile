# =============================================================================
# ==          Makefile for the CUDA Dynamical Systems Library              ==
# =============================================================================
#
# This is a generic Makefile for public distribution.
# To use it, configure the variables in the "USER CONFIGURATION" section below.
#

# =============================================================================
# ## USER CONFIGURATION ##
#
# Instructions:
#   1. Set HDF5_ROOT to the base directory of your HDF5 installation.
#   2. Set HIGHFIVE_ROOT to the base directory of your HighFive installation.
#   3. Set CUDA_ARCH to the compute capability of your GPU.
#      - Find your GPU's code here: https://developer.nvidia.com/cuda-gpus
#      - Examples: sm_86 (RTX 3080), sm_89 (RTX 4090), sm_75 (RTX 2070)
#   4. Set HOST_COMPILER to your preferred C++ host compiler.
#
# =============================================================================

HDF5_ROOT      := /usr/local/hdf5
HIGHFIVE_ROOT  := /usr/local/highfive
CUDA_ARCH      := sm_86
HOST_COMPILER  := g++

# =============================================================================
# ## COMPILER SETUP (AUTOMATIC) ##
# - No need to edit below this line for basic configuration -
# =============================================================================
# Compiler
NVCC = nvcc

# Compiler flags are built from the user-configured variables
CXXFLAGS = -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(HOST_COMPILER) -rdc=true

# Include paths
INCLUDES = -I./cuda_dynamics_lib/include \
           -I./maps \
           -I$(HIGHFIVE_ROOT)/include \
           -I$(HDF5_ROOT)/include

# Library paths and libraries to link
LDFLAGS = -L$(HDF5_ROOT)/lib -lhdf5

# =============================================================================
# ## PROJECT STRUCTURE (AUTOMATIC) ##
# =============================================================================
TARGET = dynamics_simulator

# AUTOMATICALLY find all .cu source files in the library's src directory
LIB_SRC := $(wildcard cuda_dynamics_lib/src/*.cu)

MAIN_SRC = main.cu

# This pattern rule automatically generates the list of object files (.o)
# from the list of source files (.cu) found above.
LIB_OBJ = $(LIB_SRC:.cu=.o)
MAIN_OBJ = $(MAIN_SRC:.cu=.o)

# =============================================================================
# ## BUILD RULES ##
# =============================================================================

# Define all executables you want to build
all: horton_msd horton_escape

# Rule to build the main standard map simulator
dynamics_simulator: main.cu
	@echo "==> Building Standard Map simulator"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Rule to build the Horton escape time simulator
horton_escape: main_horton_escape.cu
	@echo "==> Building Horton Escape simulator"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

horton_msd: main_horton_msd.cu
	@echo "==> Building Horton Escape simulator"
	$(NVCC) $(CXXFLAGS) $(INCLUDES)  -o $@ $< $(LDFLAGS)

# Add other rules for other executables here...

# Rule to clean up all compiled files
clean:
	@echo "==> Cleaning up build files..."
	rm -f dynamics_simulator horton_escape_simulator

# Phony targets are not files
.PHONY: all clean
