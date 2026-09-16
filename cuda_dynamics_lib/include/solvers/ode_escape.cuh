#pragma once
#include "../cuda_dynamics.h"

// =============================================================================
// == ODE Escape Time Solver Implementation
// =============================================================================
//
// max_steps can be very large (e.g. 1e7), which at ~1e6 particles means the
// full run is ~1e13 sequential RK4 evaluations. Running that as a single
// kernel launch would monopolize the GPU uninterruptibly for the whole run
// -- a poor fit for a shared machine, and any crash partway loses all
// progress. Instead the host loop below launches the kernel repeatedly in
// batches of `steps_per_batch` steps, with each particle's state persisted
// in device memory (d_state) between launches and step_offset keeping the
// physical time t = step*dt exact across the batch boundary. Particles that
// already escaped are skipped (via the d_escape_times sentinel check) in
// every later batch. This produces bit-identical results to one big launch,
// just split into resumable pieces.

template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_escape_kernel(
    SystemType system,
    ParamsType params,
    int step_offset,      // global step index this batch starts at
    int steps_this_batch, // number of steps to run in this launch
    double dt,
    long long num_particles,
    double* d_state,       // persistent per-particle state, read+written each batch
    // --- OUTPUTS ---
    double* d_escape_times, // sentinel-initialized to -1 by the host before batch 0
    double* d_escape_basins)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    if (d_escape_times[idx] != -1.0) return; // already escaped in an earlier batch

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_state[idx * DIMS + j];
    }

    for (int local_step = 0; local_step < steps_this_batch; ++local_step) {
        int global_step = step_offset + local_step;
        double t = static_cast<double>(global_step) * dt;
        rk4_step_t<DIMS, SystemType, ParamsType>(state, t, dt, system, params);
        SystemTraits<SystemType>::post_step_update(state);
        // --- GENERALIZED ESCAPE CHECK ---
        // Call the check_escape function defined in the System's Trait
        int basin_id = SystemTraits<SystemType>::check_escape(state);

        if (basin_id != 0) {
            d_escape_times[idx] = (global_step + 1) * dt;
            d_escape_basins[idx] = static_cast<double>(basin_id);
            return; // Particle escaped, exit -- no need to persist state further
        }
    }

    // Not escaped in this batch: persist state for the next launch.
    for (int j = 0; j < DIMS; ++j) {
        d_state[idx * DIMS + j] = state[j];
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
    double* h_escape_basins, // New output array
    int steps_per_batch = 20000)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    double *d_state, *d_escape_t, *d_escape_b;

    CUDA_CHECK(cudaMalloc(&d_state, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_t, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_b, num_particles * sizeof(double))); // Allocate for basins
    CUDA_CHECK(cudaMemcpy(d_state, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    // Sentinel-initialize outputs before the batch loop (the kernel used to
    // do this itself on every launch, which would have wrongly reset
    // already-escaped particles on every later batch).
    std::vector<double> h_no_escape(num_particles, -1.0);
    CUDA_CHECK(cudaMemcpy(d_escape_t, h_no_escape.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_escape_b, 0, num_particles * sizeof(double))); // 0.0 is an all-zero-bytes bit pattern

    int steps_done = 0;
    while (steps_done < max_steps) {
        int this_batch = (max_steps - steps_done < steps_per_batch) ? (max_steps - steps_done) : steps_per_batch;

        ode_escape_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
            system_functor, params, steps_done, this_batch, dt, num_particles,
            d_state, d_escape_t, d_escape_b);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        steps_done += this_batch;
    }

    CUDA_CHECK(cudaMemcpy(h_escape_times, d_escape_t, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_escape_basins, d_escape_b, num_particles * sizeof(double), cudaMemcpyDeviceToHost)); // Copy back basins

    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_escape_t));
    CUDA_CHECK(cudaFree(d_escape_b));
}
