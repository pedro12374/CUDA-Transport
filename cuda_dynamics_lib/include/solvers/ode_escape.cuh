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
    double escape_boundary_y,
    double* d_escape_times)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_initial_conditions[idx * DIMS + j];
    }

    for (int step = 0; step < max_steps; ++step) {
        double t = static_cast<double>(step) * dt;
        rk4_step_t<DIMS, SystemType, ParamsType>(state, t, dt, system, params);

        // Check for escape in the y-dimension (state[1])
        if (fabs(state[1]) >= escape_boundary_y) {
            d_escape_times[idx] = (step + 1) * dt;
            return; // Particle escaped, exit the loop for this thread
        }
    }
    d_escape_times[idx] = -1.0; // Indicate no escape
}

template <int DIMS, typename SystemType, typename ParamsType>
inline void calculate_ode_escape_time(
    const SystemType& system_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int max_steps,
    double dt,
    double escape_boundary_y,
    double* h_escape_times)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    double *d_init, *d_escape;

    CUDA_CHECK(cudaMalloc(&d_init, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_init, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    ode_escape_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
        system_functor, params, max_steps, dt, num_particles, d_init, escape_boundary_y, d_escape);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_escape_times, d_escape, num_particles * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_escape));
}