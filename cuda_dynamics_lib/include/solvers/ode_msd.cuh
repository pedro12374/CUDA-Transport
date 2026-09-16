#pragma once
#include "../cuda_dynamics.h"
#include <algorithm>
#include <cmath>

// =============================================================================
// == ODE MSD Solver Implementation
// =============================================================================
//
// Two changes from a naive "one big kernel launch, one MSD sample per step"
// implementation:
//
//  1. Batching: num_steps can be ~1e7, which at ~1e6 particles is ~1e13
//     sequential RK4 evaluations. As with the escape solver, that's split
//     into repeated kernel launches of `steps_per_batch` steps each, with
//     each particle's wrapped/unwrapped state persisted in device memory
//     between launches so the physics is unaffected -- same operations,
//     same order, just resumable and GPU-time-sliced.
//
//  2. Log-spaced MSD sampling: recording one double per step at 1e7 steps
//     is ~80MB per (A2,A3) pair. generate_log_spaced_steps() picks a much
//     smaller set of step indices (step 0, then logarithmically spaced up
//     to num_steps-1) and only those are recorded, without changing the
//     integration itself. The corresponding physical times are returned in
//     h_msd_sample_times_out so callers can save a companion time axis
//     instead of assuming uniform spacing.

// Generates a sorted, de-duplicated list of step indices in [0, num_steps-1]:
// step 0, then num_samples-1 points spaced logarithmically up to
// num_steps-1. Used to keep MSD output size independent of num_steps.
inline std::vector<int> generate_log_spaced_steps(int num_steps, int num_samples) {
    std::vector<int> steps;
    if (num_steps <= 0) return steps;
    steps.push_back(0);
    int last_step = num_steps - 1;
    if (last_step < 1) return steps;
    if (num_samples < 2) num_samples = 2;

    double log_last = std::log(static_cast<double>(last_step));
    for (int k = 0; k < num_samples; ++k) {
        double frac = static_cast<double>(k) / static_cast<double>(num_samples - 1);
        int step = static_cast<int>(std::round(std::exp(frac * log_last)));
        step = std::min(std::max(step, 1), last_step);
        steps.push_back(step);
    }
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    return steps;
}

template <int DIMS, typename SystemType, typename ParamsType>
__global__ void ode_msd_kernel(
    SystemType system,
    ParamsType params,
    int step_offset,      // global step index this batch starts at
    int steps_this_batch, // number of steps to run in this launch
    double dt,
    long long num_particles,
    const double* d_initial_conditions,
    const int* d_sample_steps, // sorted global step indices to record MSD at
    int num_samples,
    int sample_start_idx, // index into d_sample_steps this batch starts at
    double* d_state_wrapped,   // persistent per-particle state across batches
    double* d_state_unwrapped,
    double* d_total_displacement,
    double* d_displacements,
    double* d_msd)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // --- State variables ---
    double initial_state[DIMS];   // For calculating displacement
    double state_wrapped[DIMS];   // Input for the physics (RK4)
    double state_unwrapped[DIMS]; // "True" position for measurements

    for (int j = 0; j < DIMS; ++j) {
        initial_state[j] = d_initial_conditions[idx * DIMS + j];
        state_wrapped[j] = d_state_wrapped[idx * DIMS + j];
        state_unwrapped[j] = d_state_unwrapped[idx * DIMS + j];
    }

    int sample_idx = sample_start_idx;
    for (int local_step = 0; local_step < steps_this_batch; ++local_step) {
        int global_step = step_offset + local_step;
        double t = static_cast<double>(global_step) * dt;

        // --- 1. Record MSD only at the chosen log-spaced sample steps ---
        // This uses the "true" unwrapped trajectory, measured at the start
        // of the step (matches the original one-sample-per-step behavior).
        if (sample_idx < num_samples && d_sample_steps[sample_idx] == global_step) {
            double dx = state_unwrapped[0] - initial_state[0];
            double dy = state_unwrapped[1] - initial_state[1];
            atomicAdd(&d_msd[sample_idx], dx * dx + dy * dy);
            ++sample_idx;
        }

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

    // Persist state for the next launch.
    for (int j = 0; j < DIMS; ++j) {
        d_state_wrapped[idx * DIMS + j] = state_wrapped[j];
        d_state_unwrapped[idx * DIMS + j] = state_unwrapped[j];
    }

    // --- Final Displacement (recomputed every batch; only the last launch's
    //     write matters, and it's idempotent so that's harmless) ---
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
    std::vector<double>& h_msd_out,             // resized to the actual sample count
    std::vector<double>& h_msd_sample_times_out, // physical times for each h_msd_out entry
    int num_msd_samples = 300,
    int steps_per_batch = 20000)
{
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;

    std::vector<int> sample_steps = generate_log_spaced_steps(num_steps, num_msd_samples);
    int num_samples = static_cast<int>(sample_steps.size());
    h_msd_out.assign(num_samples, 0.0);
    h_msd_sample_times_out.resize(num_samples);
    for (int k = 0; k < num_samples; ++k) {
        h_msd_sample_times_out[k] = static_cast<double>(sample_steps[k]) * dt;
    }

    double *d_init, *d_state_wrapped, *d_state_unwrapped, *d_total_disp, *d_disp, *d_msd_p;
    int *d_sample_steps;

    CUDA_CHECK(cudaMalloc(&d_init, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_state_wrapped, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_state_unwrapped, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_total_disp, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_disp, num_particles * DIMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_msd_p, num_samples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sample_steps, num_samples * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_init, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_wrapped, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_unwrapped, h_initial_conditions, num_particles * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_msd_p, 0, num_samples * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sample_steps, sample_steps.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice));

    int steps_done = 0;
    while (steps_done < num_steps) {
        int this_batch = (num_steps - steps_done < steps_per_batch) ? (num_steps - steps_done) : steps_per_batch;
        int sample_start_idx = static_cast<int>(
            std::lower_bound(sample_steps.begin(), sample_steps.end(), steps_done) - sample_steps.begin());

        ode_msd_kernel<DIMS, SystemType, ParamsType><<<grid_size, block_size>>>(
            system_functor, params, steps_done, this_batch, dt, num_particles, d_init,
            d_sample_steps, num_samples, sample_start_idx,
            d_state_wrapped, d_state_unwrapped, d_total_disp, d_disp, d_msd_p);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        steps_done += this_batch;
    }

    CUDA_CHECK(cudaMemcpy(h_total_displacement, d_total_disp, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_displacements, d_disp, num_particles * DIMS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_msd_out.data(), d_msd_p, num_samples * sizeof(double), cudaMemcpyDeviceToHost));

    // Final normalization on the CPU
    for (int k = 0; k < num_samples; ++k) {
        h_msd_out[k] /= num_particles;
    }

    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_state_wrapped));
    CUDA_CHECK(cudaFree(d_state_unwrapped));
    CUDA_CHECK(cudaFree(d_total_disp));
    CUDA_CHECK(cudaFree(d_disp));
    CUDA_CHECK(cudaFree(d_msd_p));
    CUDA_CHECK(cudaFree(d_sample_steps));
}
