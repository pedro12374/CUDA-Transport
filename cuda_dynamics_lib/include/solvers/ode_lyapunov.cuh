#pragma once 
#include "../cuda_dynamics.h"

// =============================================================================
// == ODE Lyapunov Solver Implementation
// =============================================================================

template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_lyapunov_kernel(
    SystemType system,
    ParamsType params,
    int num_steps,
    double dt,
    long long num_particles,
    const double* d_initial_conditions,
    double* d_lyapunov_exp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_initial_conditions[idx * DIMS + j];
    }

    double tangent_vec[DIMS] = {1.0, 0.0}; // Initial tangent vector
    double jacobian[DIMS * DIMS];
    double sum_of_logs = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        double t = static_cast<double>(step) * dt;
        
        // 1. Evolve main trajectory
        rk4_step_t<DIMS, SystemType, ParamsType>(state, t, dt, system, params);
        
        // 2. Evolve tangent vector
        system.template jacobian<DIMS>(state, params, jacobian, t + dt);
        double new_tangent_vec[DIMS] = {0};
        matrix_vector_mult<DIMS>(jacobian, tangent_vec, new_tangent_vec);
        
        // 3. Rescale and accumulate
        double norm = vector_norm<DIMS>(new_tangent_vec);
        if (norm > 1e-12) {
            sum_of_logs += log(norm);
            normalize_vector<DIMS>(new_tangent_vec, norm);
            for(int j=0; j<DIMS; ++j) tangent_vec[j] = new_tangent_vec[j];
        }
    }

    d_lyapunov_exp[idx] = sum_of_logs / (num_steps * dt);
}


template <int DIMS, typename SystemType, typename ParamsType>
inline void calculate_ode_lyapunov_exponent(
    const SystemType& system_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_steps, // Corrected from your header
    double dt,
    double* h_lyapunov_exponents) // Corrected from your header
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;

    double *d_init_cond, *d_lyap_exp;
    CUDA_CHECK(cudaMalloc(&d_init_cond, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lyap_exp, num_particles * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_init_cond, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    ode_lyapunov_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
        system_functor, params, num_steps, dt, num_particles, d_init_cond, d_lyap_exp);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Corrected variable names below
    CUDA_CHECK(cudaMemcpy(h_lyapunov_exponents, d_lyap_exp, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_init_cond));
    CUDA_CHECK(cudaFree(d_lyap_exp));
}