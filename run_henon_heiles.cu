// Generic entry point for the Henon-Heiles system. See maps/henon_heiles.h
// for the system definition and configs/henon_heiles_*.cfg for examples.
//
// Usage: ./run_henon_heiles <config_file>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "cuda_dynamics_lib/include/runner.cuh"
#include "maps/henon_heiles.h"

int main(int argc, char** argv) {
    return run_ode_generic<4, HenonHeilesSystem, HenonHeilesParams>(argc, argv);
}
