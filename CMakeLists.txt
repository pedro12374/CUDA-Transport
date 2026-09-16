# =============================================================================
# ==            CMake build for CUDA-Transport (alternative to Makefile)   ==
# =============================================================================
#
# The Makefile remains the primary/verified build; this is offered as a
# more portable alternative (proper find_package(HDF5), CMAKE_CUDA_ARCHITECTURES,
# out-of-source builds, ctest integration).
#
# Usage:
#   mkdir build && cd build
#   cmake .. [-DCMAKE_CUDA_ARCHITECTURES=86] [-DCMAKE_BUILD_TYPE=Debug]
#   make -j
#   ctest            # or: make test / run tests/<name> individually
#
# =============================================================================

cmake_minimum_required(VERSION 3.18) # CMAKE_CUDA_ARCHITECTURES needs >= 3.18

# Must be set BEFORE project(... CUDA): CMake bakes the default -gencode
# flags in during CUDA compiler detection, triggered by project(). Setting
# this after project() silently has no effect (confirmed: without this,
# CMake 3.18/nvcc 11.2 defaults to compute_52, which lacks the native
# double-precision atomicAdd this library's MSD kernels need -- a compile
# error, not a silent wrong-architecture binary, but best avoided).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # NVIDIA A30/A100 (this dev server's GPU). Override for others, e.g.:
    #   cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 30-series
    #   cmake .. -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 40-series
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(CUDA-Transport LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Matches the Makefile's -O3 regardless of CMAKE_BUILD_TYPE.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# ---- Dependencies ----
# Found via pkg-config on this server (Debian's serial/non-MPI HDF5 package
# ships hdf5-serial.pc) -- no manual path hints needed. If find_package
# fails on another system, pass -DHDF5_ROOT=/path/to/hdf5.
find_package(HDF5 REQUIRED COMPONENTS CXX)

# ---- Shared include paths ----
set(CUDAT_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/cuda_dynamics_lib/include
    ${CMAKE_SOURCE_DIR}/maps
    ${HDF5_INCLUDE_DIRS}
)

# All library/system headers, purely so IDEs that use this CMake project
# pick them up; add_executable's dependency tracking already covers
# rebuild-on-header-change via CMake's own compiler-generated depfiles.
file(GLOB_RECURSE CUDAT_HEADERS
    ${CMAKE_SOURCE_DIR}/cuda_dynamics_lib/include/*.h
    ${CMAKE_SOURCE_DIR}/cuda_dynamics_lib/include/*.cuh
    ${CMAKE_SOURCE_DIR}/maps/*.h
)

# Defines one of the generic, config-driven per-system driver executables
# (see runner.cuh) -- mirrors the Makefile's run_<system> targets.
function(add_cudat_driver name source)
    add_executable(${name} ${source} ${CUDAT_HEADERS})
    target_include_directories(${name} PRIVATE ${CUDAT_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ${HDF5_CXX_LIBRARIES})
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endfunction()

add_cudat_driver(run_horton run_horton.cu)
add_cudat_driver(run_standard_map run_standard_map.cu)
add_cudat_driver(run_henon_heiles run_henon_heiles.cu)

# ---- Test suite (tests/test_*.cu) ----
enable_testing()
file(GLOB CUDAT_TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/test_*.cu)
foreach(test_src ${CUDAT_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE ${CUDAT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(${test_name} PRIVATE ${HDF5_CXX_LIBRARIES})
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
