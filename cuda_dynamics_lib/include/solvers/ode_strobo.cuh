#pragma once
/**
 * @file ode_strobo.cuh
 * @brief Stroboscopic map solver for continuous-time systems -- samples
 * each trajectory at fixed intervals `tau` rather than every dt, the
 * standard way to turn a periodically- (or quasi-periodically-) forced
 * flow into a Poincare-section-like map: calculate_ode_stroboscopic_map().
 * Normally called via `calculation = stroboscopic` in a config file (see
 * run_ode_generic() in runner.cuh) rather than directly.
 */
#include "../cuda_dynamics.h"

// =============================================================================
// == ODE Stroboscopic Map Solver Implementation
// =============================================================================

template <int DIMS, typename SystemType, typename ParamsType>
__device__ void integrate_for_tau_device(
    double state[DIMS],
    double start_time,
    double tau,
    double dt,
    const SystemType& system,
    const ParamsType& params)
{
    int num_steps = static_cast<int>(tau / dt);
    for (int i = 0; i < num_steps; ++i) {
        double current_t = start_time + static_cast<double>(i) * dt;
        rk4_step_t<DIMS, SystemType, ParamsType>(state, current_t, dt, system, params);
    }

    // tau isn't generally an exact multiple of dt, so the loop above lands
    // short of tau by a remainder in [0, dt). Take one final partial-dt
    // step to land exactly on tau -- otherwise each interval's state would
    // silently fall further and further behind the nominal p*tau clock
    // used to seed the next interval, desyncing the time-dependent forcing
    // terms from the true integrated time.
    double elapsed = static_cast<double>(num_steps) * dt;
    double remaining = tau - elapsed;
    if (remaining > 1e-12 * dt) {
        double current_t = start_time + elapsed;
        rk4_step_t<DIMS, SystemType, ParamsType>(state, current_t, remaining, system, params);
    }
}


template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_stroboscopic_kernel(
    SystemType system,
    ParamsType params,
    int num_points, // How many points to record per trajectory
    double tau,     // The time interval between points
    double dt,      // The integration timestep
    long long num_particles,
    const double* d_initial_conditions,
    double* d_stroboscopic_map_out) // Output array
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double state[DIMS];
    for (int j = 0; j < DIMS; ++j) {
        state[j] = d_initial_conditions[idx * DIMS + j];
    }

    // Loop to generate each point in the stroboscopic map
    for (int p = 0; p < num_points; ++p) {
        // 1. Save the current state to the output array
        long long flat_idx = (idx * num_points + p) * DIMS;
        for (int j = 0; j < DIMS; ++j) {
            d_stroboscopic_map_out[flat_idx + j] = state[j];
        }

        // 2. Integrate the trajectory forward by one 'tau' interval
        double trajectory_start_time = static_cast<double>(p) * tau;
        integrate_for_tau_device<DIMS, SystemType, ParamsType>(
            state, trajectory_start_time, tau, dt, system, params);
    }
}


/**
 * @brief Records each particle's state every `tau` time units (not every
 * `dt`), for `num_points` intervals -- a stroboscopic/Poincare-section-like
 * map of a continuous-time trajectory.
 *
 * @tparam DIMS State dimensionality.
 * @tparam SystemType A system type implementing the ODE interface (see maps/horton.h).
 * @tparam ParamsType That system's parameter struct type.
 * @param system_functor The system functor.
 * @param params Physical parameters.
 * @param h_initial_conditions Host array, `num_particles * DIMS` doubles.
 * @param num_particles Particle count.
 * @param num_points Number of stroboscopic samples to record per particle.
 * @param tau Time interval between samples.
 * @param dt Integration step size (need not divide `tau` evenly; see
 * integrate_for_tau_device()).
 * @param h_stroboscopic_map_out [out] Host array, `num_particles *
 * num_points * DIMS` doubles, layout `(particle, sample, dimension)`.
 */
template <int DIMS, typename SystemType, typename ParamsType>
inline void calculate_ode_stroboscopic_map(
    const SystemType& system_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_points,
    double tau,
    double dt,
    double* h_stroboscopic_map_out)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    double *d_init_cond, *d_strobo_map;
    size_t map_size = num_particles * num_points * DIMS * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_init_cond, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_strobo_map, map_size));
    CUDA_CHECK(cudaMemcpy(d_init_cond, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));

    ode_stroboscopic_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
        system_functor, params, num_points, tau, dt, num_particles, d_init_cond, d_strobo_map);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_stroboscopic_map_out, d_strobo_map, map_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_init_cond));
    CUDA_CHECK(cudaFree(d_strobo_map));
}