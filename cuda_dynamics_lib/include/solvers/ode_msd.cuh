#pragma once 
#include "../cuda_dynamics.h"

// =============================================================================
// == ODE MSD Solver Implementation
// =============================================================================

template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_msd_kernel(
    SystemType system,
    ParamsType params,
    int num_steps,
    double dt,
    long long num_particles,
    const double* d_initial_conditions,
    double* d_total_displacement,
    double* d_displacements,
    double* d_msd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // --- State variables ---
    double state_wrapped[DIMS];   // Input for the physics (RK4)
    double state_unwrapped[DIMS]; // "True" position for measurements
    double initial_state[DIMS];   // For calculating displacement

    for (int j = 0; j < DIMS; ++j) {
        initial_state[j] = d_initial_conditions[idx * DIMS + j];
        state_wrapped[j] = initial_state[j];
        state_unwrapped[j] = initial_state[j];
    }

    for (int step = 0; step < num_steps; ++step) {
        double t = static_cast<double>(step) * dt;

        // --- 1. Calculate MSD for BOTH directions at the start of the step ---
        // This uses the "true" unwrapped trajectory.
        double dx = state_unwrapped[0] - initial_state[0];
        double dy = state_unwrapped[1] - initial_state[1];
        atomicAdd(&d_msd[step], dx * dx + dy * dy);


        double state_old_wrapped[DIMS];
        for (int j = 0; j < DIMS; ++j) {
            state_old_wrapped[j] = state_wrapped[j];
        }
        // --- 2. Evolve the system ---
        // The integrator takes the WRAPPED state as input and updates it
        // in-place to the next UNWRAPPED position.
        rk4_step_t<DIMS, SystemType, ParamsType>(state_wrapped, t, dt, system, params);
        
        for (int j = 0; j < DIMS; ++j) {
            double step_displacement = state_wrapped[j] - state_old_wrapped[j];
            state_unwrapped[j] += step_displacement;
        }

        // --- 3. Update the unwrapped state ---
        // The result of the integration IS the new unwrapped state.


        // --- 4. Create the new wrapped state for the NEXT iteration's physics ---
        // The SystemTraits function now applies periodicity ONLY to the wrapped state.
        SystemTraits<SystemType>::post_step_update(state_wrapped);
    }
    
    // --- Final Displacement (calculated from the final unwrapped state) ---
    double final_dx = state_unwrapped[0] - initial_state[0];
    double final_dy = state_unwrapped[1] - initial_state[1];
    d_displacements[idx * DIMS + 0] = final_dx;
    d_displacements[idx * DIMS + 1] = final_dy;
    d_total_displacement[idx] = sqrt(final_dx * final_dx + final_dy * final_dy);
}



template <int DIMS, typename SystemType, typename ParamsType>
inline void calculate_ode_msd_and_displacement(
    const SystemType& system_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_steps,
    double dt,
    double* h_total_displacement,
    double* h_displacements,
    double* h_msd)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    double *d_init, *d_total_disp, *d_disp, *d_msd_p;
    
    CUDA_CHECK(cudaMalloc(&d_init, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_total_disp, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_disp, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_msd_p, num_steps * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_init, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_msd_p, 0, num_steps * sizeof(double)));

    // Corrected kernel launch: use 'num_steps' instead of 'max_steps'
    ode_msd_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
        system_functor, params, num_steps, dt, num_particles, d_init, d_total_disp, d_disp, d_msd_p);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_total_displacement, d_total_disp, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_displacements, d_disp, num_particles * DIMS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_msd, d_msd_p, num_steps * sizeof(double), cudaMemcpyDeviceToHost));

    // Final normalization on the CPU
    for (int i = 0; i < num_steps; ++i) {
        h_msd[i] /= num_particles;
    }

    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_total_disp));
    CUDA_CHECK(cudaFree(d_disp));
    CUDA_CHECK(cudaFree(d_msd_p));
}