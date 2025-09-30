#pragma once 
#include "../cuda_dynamics.h"

// =============================================================================
// == CUDA KERNEL for Escape Time and Basins
// =============================================================================
template <int DIMS, typename MapType, typename ParamsType>
__global__ void escape_time_kernel_generic(
    MapType map,
    ParamsType params,
    int max_iterations,
    long long num_particles,
    const double* d_initial_conditions,
    // Output arrays
    double* d_escape_times,
    double* d_escape_basins)
{
    // 1D Thread Indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to process all particles
    for (long long i = idx; i < num_particles; i += stride) {
        // --- State variables for the trajectory ---
        double state_map[DIMS];
        for (int j = 0; j < DIMS; ++j) {
            state_map[j] = d_initial_conditions[i * DIMS + j];
        }

        // --- Initialization ---
        bool escaped = false;
        double escape_time = -1.0; // -1 indicates no escape
        double escape_basin = 0.0; // 0 indicates no escape

        // --- Main Integration Loop ---
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (!escaped) {
                // Evolve the trajectory by one step
                map.template operator()<DIMS>(state_map, nullptr, params);

                // Check for escape conditions using MapTraits
                escape_basin = MapTraits<MapType>::check_escape(state_map);

                if (escape_basin != 0.0) {
                    escaped = true;
                    escape_time = static_cast<double>(iter)+1.0;
                }
            }
        }

        // --- Store Results ---
        d_escape_times[i] = escape_time;
        d_escape_basins[i] = escape_basin;
    }
}

// =============================================================================
// == HOST SOLVER FUNCTION for Escape Time
// =============================================================================
template <int DIMS, typename MapType, typename ParamsType>
inline void calculate_escape_time(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int max_iterations,
    // Output arrays (host pointers)
    double* h_escape_times,
    double* h_escape_basins)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;

    double *d_init_cond, *d_escape_times, *d_escape_basins;
    CUDA_CHECK(cudaMalloc(&d_init_cond, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_times, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_escape_basins, num_particles * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_init_cond, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    escape_time_kernel_generic<DIMS, MapType, ParamsType><<<grid_size, block_size>>>(
        map_functor, params, max_iterations, num_particles, d_init_cond,
        d_escape_times, d_escape_basins);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_escape_times, d_escape_times, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_escape_basins, d_escape_basins, num_particles * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_init_cond));
    CUDA_CHECK(cudaFree(d_escape_times));
    CUDA_CHECK(cudaFree(d_escape_basins));
}