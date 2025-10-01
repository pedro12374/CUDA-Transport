#pragma once 
#include "../cuda_dynamics.h"

// =============================================================================
// == ODE Escape Time Solver Implementation
// =============================================================================


template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_escape_kernel(
    SystemType system,
    ParamsType params,
    int max_steps,
    double dt,
    long long num_particles,
    const double* d_initial_conditions,
    // --- OUTPUTS ---
    double* d_escape_times,
    double* d_escape_basins) // New output array for basin IDs
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_initial_conditions[idx * DIMS + j];
    }

    // Initialize outputs
    d_escape_times[idx] = -1.0; // No escape
    d_escape_basins[idx] = 0.0;  // Basin 0 (no escape)

    for (int step = 0; step < max_steps; ++step) {
        double t = static_cast<double>(step) * dt;
        rk4_step_t<DIMS, SystemType, ParamsType>(state, t, dt, system, params);
         SystemTraits<SystemType>::post_step_update(state);
        // --- GENERALIZED ESCAPE CHECK ---
        // Call the check_escape function defined in the System's Trait
        int basin_id = SystemTraits<SystemType>::check_escape(state);

        if (basin_id != 0) {
            d_escape_times[idx] = (step + 1) * dt;
            d_escape_basins[idx] = static_cast<double>(basin_id);
            return; // Particle escaped, exit
        }
    }
}

template <int DIMS, typename SystemType, typename ParamsType>
inline void calculate_ode_escape( // Renamed for clarity
    const SystemType& system_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int max_steps,
    double dt,
    // --- OUTPUTS ---
    double* h_escape_times,
    double* h_escape_basins) // New output array
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    double *d_init, *d_escape_t, *d_escape_b;

    CUDA_CHECK(cudaMalloc(&d_init, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_t, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_b, num_particles * sizeof(double))); // Allocate for basins
    CUDA_CHECK(cudaMemcpy(d_init, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    ode_escape_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
        system_functor, params, max_steps, dt, num_particles, d_init, 
        d_escape_t, d_escape_b);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_escape_times, d_escape_t, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_escape_basins, d_escape_b, num_particles * sizeof(double), cudaMemcpyDeviceToHost)); // Copy back basins

    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_escape_t));
    CUDA_CHECK(cudaFree(d_escape_b));
}