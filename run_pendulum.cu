// Generic entry point for the driven, damped pendulum (see TUTORIAL.md and
// maps/pendulum.h). Usage: ./run_pendulum <config_file>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "cuda_dynamics_lib/include/runner.cuh"
#include "maps/pendulum.h"

int main(int argc, char** argv) {
    return run_ode_generic<2, PendulumSystem, PendulumParams>(argc, argv);
}
