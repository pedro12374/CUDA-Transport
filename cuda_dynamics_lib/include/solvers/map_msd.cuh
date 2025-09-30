#pragma once 
#include "../cuda_dynamics.h"

template <int DIMS, typename MapType, typename ParamsType>
__global__ void msd_kernel_generic(
    MapType map,
    ParamsType params, // Passed by value
    int num_iterations,
    long long num_particles,
    const double* d_initial_conditions,
    // Note: min/max bounds are no longer needed by this generic kernel,
    // but a specific map might need them passed via its Params struct.
    // We remove them from the kernel signature for true generality.
    double* d_total_displacement,
    double* d_displacements,
    double* d_msd)
{
    // 1D Thread Indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to process all particles
    for (long long i = idx; i < num_particles; i += stride) {
        // state_map holds the wrapped coordinates for the map's dynamics
        // state_unwrapped holds the true coordinates for measurement
        double state_map[DIMS], state_unwrapped[DIMS], state0[DIMS];

        // Initialize all states to the particle's initial condition
        for (int j = 0; j < DIMS; ++j) {
            state0[j] = d_initial_conditions[i * DIMS + j];
            state_map[j] = state0[j];
            state_unwrapped[j] = state0[j];
        }

        for (int iter = 0; iter < num_iterations; ++iter) {
            // --- GENERIC MAP APPLICATION ---
            // The map functor is now responsible for updating both the wrapped
            // and unwrapped state arrays according to its specific physics.
            map.operator()<DIMS>(state_map, state_unwrapped, params);

            // --- GENERIC MSD CALCULATION ---
            // Use the MapTraits to identify which dimension to measure for MSD.
            const int msd_dim = MapTraits<MapType>::msd_dimension_index;
            double disp = state_unwrapped[msd_dim] - state0[msd_dim];
            atomicAdd(&d_msd[iter], disp * disp);
        }

        // --- Final Displacement Calculation ---
        double final_total_sq_disp = 0.0;
        for (int j = 0; j < DIMS; ++j) {
            double disp_j = state_unwrapped[j] - state0[j];
            d_displacements[i * DIMS + j] = disp_j;
            final_total_sq_disp += disp_j * disp_j;
        }
        d_total_displacement[i] = sqrt(final_total_sq_disp);
    }
}

template <int DIMS, typename MapType, typename ParamsType>
inline void calculate_msd_and_displacement(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    const double* min_bounds, // Kept in signature for compatibility with main.cu
    const double* max_bounds, // but no longer used in this generic solver.
    long long num_particles,
    int num_iterations,
    double* h_total_displacement,
    double* h_displacements,
    double* h_msd)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    dim3 block_dim(block_size);
    dim3 grid_dim(grid_size);

    double *d_init_cond, *d_total_disp, *d_displacements, *d_msd;
    
    CUDA_CHECK(cudaMalloc(&d_init_cond, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_total_disp, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_displacements, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_msd, num_iterations * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_init_cond, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_msd, 0, num_iterations * sizeof(double)));

    // Launch the new kernel, which no longer requires boundary information.
    msd_kernel_generic<DIMS, MapType, ParamsType><<<grid_dim, block_dim>>>(
        map_functor, params, num_iterations, num_particles, d_init_cond,
        d_total_disp, d_displacements, d_msd);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_total_displacement, d_total_disp, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_displacements, d_displacements, num_particles * DIMS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_msd, d_msd, num_iterations * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_iterations; ++i) {
        h_msd[i] /= num_particles;
    }

    CUDA_CHECK(cudaFree(d_init_cond));
    CUDA_CHECK(cudaFree(d_total_disp));
    CUDA_CHECK(cudaFree(d_displacements));
    CUDA_CHECK(cudaFree(d_msd));
}
