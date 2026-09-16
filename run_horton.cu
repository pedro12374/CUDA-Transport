// Generic entry point for the Horton (three-wave drift) system. All the
// actual work -- grid setup, parameter sweeps, which calculation to run,
// HDF5 output -- is driven by a config file (see configs/horton_*.cfg) and
// handled by run_ode_generic (cuda_dynamics_lib/include/runner.cuh).
//
// Usage: ./run_horton <config_file>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "cuda_dynamics_lib/include/runner.cuh"
#include "maps/horton.h"

int main(int argc, char** argv) {
    return run_ode_generic<2, HortonSystem, HortonSystemParams>(argc, argv);
}
