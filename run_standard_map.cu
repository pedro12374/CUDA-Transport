// Generic entry point for the Standard Map. All the actual work -- grid
// setup, parameter sweeps, which calculation to run, HDF5 output -- is
// driven by a config file (see configs/standard_map_*.cfg) and handled by
// run_map_generic (cuda_dynamics_lib/include/runner.cuh).
//
// Usage: ./run_standard_map <config_file>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "cuda_dynamics_lib/include/runner.cuh"
#include "maps/standard_map.h"

int main(int argc, char** argv) {
    return run_map_generic<2, StandardMap, StandardMapParams>(argc, argv);
}
