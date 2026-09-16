// Regression test for the batched ode_escape/ode_msd solvers (see
// CLAUDE.md's Phase 2 notes): splitting a run into many small kernel
// launches (steps_per_batch) must produce the same result as one huge
// launch. Originally validated ad hoc during Phase 2; formalized here so
// future changes to the batching logic can't silently break it.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cuda_dynamics.h"
#include "horton.h"
#include "henon_heiles.h"
#include "common.h"

template <int DIMS, typename SystemType, typename ParamsType>
static bool check_escape_batching(const char* name, const SystemType& system, const ParamsType& params,
                                   const std::vector<double>& ics, long long n,
                                   int max_steps, double dt) {
    std::vector<double> t_big(n), b_big(n), t_small(n), b_small(n);
    calculate_ode_escape<DIMS, SystemType, ParamsType>(
        system, params, ics.data(), n, max_steps, dt, t_big.data(), b_big.data(), 1000000);
    calculate_ode_escape<DIMS, SystemType, ParamsType>(
        system, params, ics.data(), n, max_steps, dt, t_small.data(), b_small.data(), 7);

    int mismatches = 0, escaped = 0;
    for (long long i = 0; i < n; ++i) {
        if (t_big[i] != -1.0) ++escaped;
        if (t_big[i] != t_small[i] || b_big[i] != b_small[i]) ++mismatches;
    }
    printf("         %s escape: %lld particles, %d escaped, %d mismatches (batch=1e6 vs batch=7)\n",
           name, n, escaped, mismatches);
    return check(mismatches == 0, std::string(name) + " escape batching mismatch")
        && check(escaped > 0, std::string(name) + " escape batching test exercised no escapes");
}

template <int DIMS, typename SystemType, typename ParamsType>
static bool check_msd_batching(const char* name, const SystemType& system, const ParamsType& params,
                                const std::vector<double>& ics, long long n,
                                int num_steps, double dt, int num_samples) {
    std::vector<double> td_big(n), disp_big(n * DIMS), td_small(n), disp_small(n * DIMS);
    std::vector<double> msd_big, t_big, msd_small, t_small;

    calculate_ode_msd_and_displacement<DIMS, SystemType, ParamsType>(
        system, params, ics.data(), n, num_steps, dt, td_big.data(), disp_big.data(),
        msd_big, t_big, num_samples, 1000000);
    calculate_ode_msd_and_displacement<DIMS, SystemType, ParamsType>(
        system, params, ics.data(), n, num_steps, dt, td_small.data(), disp_small.data(),
        msd_small, t_small, num_samples, 13); // 13 shares no common factor with num_steps

    double max_disp_diff = 0.0;
    for (size_t i = 0; i < disp_big.size(); ++i)
        max_disp_diff = std::max(max_disp_diff, std::fabs(disp_big[i] - disp_small[i]));
    double max_msd_diff = 0.0;
    for (size_t i = 0; i < msd_big.size(); ++i)
        max_msd_diff = std::max(max_msd_diff, std::fabs(msd_big[i] - msd_small[i]));
    bool times_match = (t_big.size() == t_small.size());
    for (size_t i = 0; times_match && i < t_big.size(); ++i)
        if (t_big[i] != t_small[i]) times_match = false;

    printf("         %s msd: %zu samples, max |disp diff| = %.3e, max |msd diff| = %.3e, times match = %s\n",
           name, msd_big.size(), max_disp_diff, max_msd_diff, times_match ? "yes" : "no");

    // displacement is per-thread (no atomics) so must match exactly; msd is
    // accumulated via atomicAdd, so a different batch schedule can reorder
    // the floating-point sum -- allow FP-noise-level (not zero) tolerance.
    return check(max_disp_diff == 0.0, std::string(name) + " msd batching: displacement mismatch")
        && check(max_msd_diff < 1e-9, std::string(name) + " msd batching: MSD value mismatch")
        && check(times_match, std::string(name) + " msd batching: sample time mismatch");
}

static bool test_horton_batching() {
    HortonSystem system;
    HortonSystemParams params = {
        .A1 = 1.0, .A2 = 0.5, .A3 = 0.3,
        .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
        .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
        .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
    };
    params.v2 = std::fabs(params.w2 / params.ky2 - params.w1 / params.ky1);
    params.v3 = std::fabs(params.w3 / params.ky3 - params.w1 / params.ky1);

    GridSetup grid(2, {12, 12}, {-M_PI, -2 * M_PI}, {M_PI, 2 * M_PI});

    bool ok = check_escape_batching<2>("Horton", system, params, grid.h_initial_conditions,
                                        grid.num_particles, 500, 0.01);
    ok &= check_msd_batching<2>("Horton", system, params, grid.h_initial_conditions,
                                 grid.num_particles, 800, 0.01, 40);
    return ok;
}

static bool test_henon_heiles_batching() {
    HenonHeilesSystem system;
    HenonHeilesParams params = { .lambda = 1.0 };

    // Small grid released from rest, spanning confined and escaping energies.
    GridSetup grid(4, {10, 10, 1, 1}, {-1.0, -1.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0});

    bool ok = check_escape_batching<4>("Henon-Heiles", system, params, grid.h_initial_conditions,
                                        grid.num_particles, 2000, 0.01);
    ok &= check_msd_batching<4>("Henon-Heiles", system, params, grid.h_initial_conditions,
                                 grid.num_particles, 800, 0.01, 40);
    return ok;
}

int main() {
    TestRunner t;
    t.run("horton_batching", test_horton_batching);
    t.run("henon_heiles_batching", test_henon_heiles_batching);
    return t.summary();
}
