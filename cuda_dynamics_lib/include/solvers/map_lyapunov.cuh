#pragma once 
#include "../cuda_dynamics.h"

// =============================================================================
// == CUDA KERNEL for Maximum Lyapunov Exponent
// =============================================================================
template <int DIMS, typename MapType, typename ParamsType>
__global__ void lyapunov_kernel_generic(
    MapType map,
    ParamsType params,
    int num_iterations,
    long long num_particles,
    const double* d_initial_conditions,
    // Output array
    double* d_lyapunov_exp)
{
    // 1D Thread Indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to process all particles
    for (long long i = idx; i < num_particles; i += stride) {
        // --- State variables for the main trajectory ---
        double state_map[DIMS];
        for (int j = 0; j < DIMS; ++j) {
            state_map[j] = d_initial_conditions[i * DIMS + j];
        }

        // --- State variables for the tangent vector ---
        double tangent_vector[DIMS];
        for (int j = 0; j < DIMS; ++j) {
            tangent_vector[j] = (j == 0) ? 1.0 : 0.0; // Start with a basis vector
        }
        double jacobian[DIMS * DIMS];
        double sum_of_logs = 0.0;

        for (int iter = 0; iter < num_iterations; ++iter) {
            // 1. Evolve the main trajectory one step
            map.operator()<DIMS>(state_map, nullptr, params); // Pass nullptr for unwrapped state

            // 2. Calculate the Jacobian matrix at the new point
            // CORRECTED: Explicitly provide the template argument <DIMS>
            map.jacobian<DIMS>(state_map, params, jacobian);

            // 3. Evolve the tangent vector by multiplying with the Jacobian
            double new_tangent_vector[DIMS] = {0};
            for (int row = 0; row < DIMS; ++row) {
                for (int col = 0; col < DIMS; ++col) {
                    new_tangent_vector[row] += jacobian[row * DIMS + col] * tangent_vector[col];
                }
            }

            // 4. Calculate the norm (length) of the new tangent vector
            double norm = 0.0;
            for (int j = 0; j < DIMS; ++j) {
                norm += new_tangent_vector[j] * new_tangent_vector[j];
            }
            norm = sqrt(norm);

            // 5. Add the log of the norm to our running sum
            if (norm > 0) { // Avoid log(0)
                sum_of_logs += log(norm);
            }

            // 6. Renormalize the tangent vector for the next iteration
            for (int j = 0; j < DIMS; ++j) {
                tangent_vector[j] = new_tangent_vector[j] / norm;
            }
        }

        // --- Final Calculation ---
        // The Lyapunov exponent is the average of the logs
        d_lyapunov_exp[i] = sum_of_logs / num_iterations;
    }
}


// =============================================================================
// == HOST SOLVER FUNCTION for Lyapunov Exponent
// =============================================================================
template <int DIMS, typename MapType, typename ParamsType>
inline void calculate_lyapunov_exponent(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_iterations,
    double* h_lyapunov_exp)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    dim3 block_dim(block_size);
    dim3 grid_dim(grid_size);

    double *d_init_cond, *d_lyap_exp;
    CUDA_CHECK(cudaMalloc(&d_init_cond, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lyap_exp, num_particles * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_init_cond, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_lyap_exp, 0, num_particles * sizeof(double)));

    lyapunov_kernel_generic<DIMS, MapType, ParamsType><<<grid_size, block_dim>>>(
        map_functor, params, num_iterations, num_particles, d_init_cond, d_lyap_exp);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_lyapunov_exp, d_lyap_exp, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_init_cond));
    CUDA_CHECK(cudaFree(d_lyap_exp));
}