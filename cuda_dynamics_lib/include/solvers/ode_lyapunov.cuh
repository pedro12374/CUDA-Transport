#pragma once
/**
 * @file ode_lyapunov.cuh
 * @brief Maximum Lyapunov exponent solver for continuous-time systems:
 * calculate_ode_lyapunov_exponent(). Normally called via
 * `calculation = lyapunov` in a config file (see run_ode_generic() in
 * runner.cuh) rather than directly.
 *
 * @note Unlike ode_escape.cuh/ode_msd.cuh, this solver runs the entire
 * `num_steps` loop in one kernel launch -- not batched. Fine for the
 * iteration counts a Lyapunov exponent typically needs to converge, but
 * see those two files' comments for why a very long run would want
 * batching (not implemented here; no current driver needs it).
 */
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
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_initial_conditions[idx * DIMS + j];
    }

    // Initial tangent vector: all-ones (normalized below), not a coordinate
    // basis vector like (1,0,...,0). The starting direction doesn't affect
    // the converged Lyapunov exponent for a generic trajectory -- except
    // when it lies exactly in an invariant subspace of the linearized
    // dynamics, which a basis vector can: e.g. Henon-Heiles' Jacobian
    // decouples into an (x,px) block and a (y,py) block whenever a
    // trajectory sits on the x=px=0 symmetric orbit (a common initial
    // condition, e.g. y-py Poincare-section grids), so (1,0,0,0) would stay
    // confined to the (x,px) block forever. For Henon-Heiles specifically
    // that block happens to carry the equal-or-larger exponent (the (y,py)
    // block alone is a bound 1-DOF system, always exactly integrable, so
    // its own confined exponent is 0), so this didn't produce a
    // demonstrably wrong answer there -- but relying on that coincidence
    // isn't sound for other systems/symmetries, so this is fixed generally.
    double tangent_vec[DIMS];
    for (int j = 0; j < DIMS; ++j) tangent_vec[j] = 1.0;
    normalize_vector<DIMS>(tangent_vec, vector_norm<DIMS>(tangent_vec));
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


/**
 * @brief Computes the maximum Lyapunov exponent for every particle:
 * integrates the trajectory with RK4 while evolving a tangent vector
 * through the system's jacobian(), periodically renormalizing and
 * accumulating `log(norm)` (the standard method).
 *
 * @tparam DIMS State dimensionality.
 * @tparam SystemType A system type implementing the ODE interface,
 * including jacobian() (see maps/horton.h).
 * @tparam ParamsType That system's parameter struct type.
 * @param system_functor The system functor.
 * @param params Physical parameters.
 * @param h_initial_conditions Host array, `num_particles * DIMS` doubles.
 * @param num_particles Particle count.
 * @param num_steps Integration steps.
 * @param dt Integration step size.
 * @param h_lyapunov_exponents [out] Host array, `num_particles` doubles.
 */
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