// GPU vs CPU reference trajectories, for every system in the library.
//
// For the ODE systems (Horton, Henon-Heiles), the real GPU stroboscopic
// solver is run with tau == dt so every RK4 step is recorded (this also
// keeps clear of the ode_strobo tau/dt-not-a-multiple case, not what's
// being tested here), then compared against tests/common.h's independently
// re-typed cpu_rk4_step.
//
// For the discrete map (Standard Map), the real GPU escape-time kernel is
// compared against a host-side loop calling the same map functor directly
// (the functor is __host__ __device__, so this exercises genuinely
// separate code paths -- host arithmetic vs the actual kernel/threading --
// not the same instructions re-run on a different processor).
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cuda_dynamics.h"
#include "horton.h"
#include "henon_heiles.h"
#include "standard_map.h"
#include "../tests/common.h"

static bool test_horton_gpu_vs_cpu() {
    const int DIMS = 2;
    const double DT = 0.01;
    const int NUM_POINTS = 50;

    HortonSystem system;
    HortonSystemParams params = {
        .A1 = 1.0, .A2 = 0.5, .A3 = 0.3,
        .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
        .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
        .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
    };
    derive_horton_velocities(params);

    std::vector<double> ics = {
        0.1, 0.2,   -1.5, 2.0,   2.8, -5.0,   0.0, 0.0,
        -2.9, 5.9,   1.234, -3.456,   3.0, -6.2,   -0.5, 4.1,
    };
    long long n = ics.size() / DIMS;

    std::vector<double> gpu_traj(n * NUM_POINTS * DIMS);
    calculate_ode_stroboscopic_map<DIMS, HortonSystem, HortonSystemParams>(
        system, params, ics.data(), n, NUM_POINTS, DT, DT, gpu_traj.data());

    double max_diff = 0.0;
    for (long long p = 0; p < n; ++p) {
        double state[DIMS] = { ics[p * DIMS], ics[p * DIMS + 1] };
        for (int pt = 0; pt < NUM_POINTS; ++pt) {
            long long gi = (p * NUM_POINTS + pt) * DIMS;
            max_diff = std::max(max_diff, std::fabs(gpu_traj[gi + 0] - state[0]));
            max_diff = std::max(max_diff, std::fabs(gpu_traj[gi + 1] - state[1]));
            cpu_rk4_step<DIMS, HortonSystem, HortonSystemParams>(state, pt * DT, DT, system, params);
        }
    }
    printf("         Horton: %lld particles x %d points, max |GPU-CPU| = %.3e\n", n, NUM_POINTS, max_diff);
    return check(max_diff < 1e-9, "Horton GPU vs CPU trajectory difference");
}

static bool test_henon_heiles_gpu_vs_cpu() {
    const int DIMS = 4;
    const double DT = 0.01;
    const int NUM_POINTS = 50;

    HenonHeilesSystem system;
    HenonHeilesParams params = { .lambda = 1.0 };

    // A mix of low-energy (confined) and higher-energy (near/above escape
    // threshold) initial conditions, released from rest (px0=py0=0).
    std::vector<double> ics = {
        0.1, 0.1, 0.0, 0.0,
        0.3, -0.2, 0.0, 0.0,
        -0.4, 0.3, 0.0, 0.0,
        0.05, -0.5, 0.0, 0.0,
        0.6, 0.1, 0.0, 0.0,
    };
    long long n = ics.size() / DIMS;

    std::vector<double> gpu_traj(n * NUM_POINTS * DIMS);
    calculate_ode_stroboscopic_map<DIMS, HenonHeilesSystem, HenonHeilesParams>(
        system, params, ics.data(), n, NUM_POINTS, DT, DT, gpu_traj.data());

    double max_diff = 0.0;
    for (long long p = 0; p < n; ++p) {
        double state[DIMS];
        for (int j = 0; j < DIMS; ++j) state[j] = ics[p * DIMS + j];
        for (int pt = 0; pt < NUM_POINTS; ++pt) {
            long long gi = (p * NUM_POINTS + pt) * DIMS;
            for (int j = 0; j < DIMS; ++j) {
                max_diff = std::max(max_diff, std::fabs(gpu_traj[gi + j] - state[j]));
            }
            cpu_rk4_step<DIMS, HenonHeilesSystem, HenonHeilesParams>(state, pt * DT, DT, system, params);
        }
    }
    printf("         Henon-Heiles: %lld particles x %d points, max |GPU-CPU| = %.3e\n", n, NUM_POINTS, max_diff);
    return check(max_diff < 1e-9, "Henon-Heiles GPU vs CPU trajectory difference");
}

static bool test_standard_map_gpu_vs_cpu() {
    const int DIMS = 2;
    const int ITERATIONS = 200;

    StandardMap map;
    // K deliberately kept below the chaos threshold (K_c ~ 0.9716). Above
    // it, momentum random-walks unboundedly (see CLAUDE.md/map_escape.cuh);
    // a few ULP of difference between separately-compiled host and device
    // code -- entirely benign -- gets exponentially amplified by the
    // positive Lyapunov exponent into a different escape time after enough
    // iterations. That's real chaos, not a bug, but it makes a strict
    // per-particle escape-time comparison meaningless (confirmed
    // empirically: K=1.5 shows mismatches, K<=0.8 doesn't). Chaotic-regime
    // behavior is exercised by test_confinement.cu instead, which only
    // checks aggregate escape/no-escape, not individual escape times.
    StandardMapParams params = { .K = 0.5 };

    std::vector<double> ics = {
        0.1, 0.1,   3.0, 0.5,   -2.0, 4.0,   0.01, 6.2,   1.5707963, 3.14159,
    };
    long long n = ics.size() / DIMS;

    std::vector<double> gpu_times(n), gpu_basins(n);
    calculate_escape_time<DIMS, StandardMap, StandardMapParams>(
        map, params, ics.data(), n, ITERATIONS, gpu_times.data(), gpu_basins.data());

    int mismatches = 0;
    for (long long p = 0; p < n; ++p) {
        double state[DIMS] = { ics[p * DIMS], ics[p * DIMS + 1] };
        double cpu_time = -1.0, cpu_basin = 0.0;
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            map.template operator()<DIMS>(state, nullptr, params);
            double basin = MapTraits<StandardMap>::check_escape(state);
            if (basin != 0.0) {
                cpu_time = static_cast<double>(iter) + 1.0;
                cpu_basin = basin;
                break;
            }
        }
        if (cpu_time != gpu_times[p] || cpu_basin != gpu_basins[p]) {
            ++mismatches;
            printf("         mismatch particle %lld: GPU(t=%.1f,b=%.1f) CPU(t=%.1f,b=%.1f)\n",
                   p, gpu_times[p], gpu_basins[p], cpu_time, cpu_basin);
        }
    }
    printf("         Standard Map: %lld particles, %d mismatches\n", n, mismatches);
    return check(mismatches == 0, "Standard Map GPU vs CPU escape time/basin mismatches");
}

int main() {
    TestRunner t;
    t.run("horton_gpu_vs_cpu", test_horton_gpu_vs_cpu);
    t.run("henon_heiles_gpu_vs_cpu", test_henon_heiles_gpu_vs_cpu);
    t.run("standard_map_gpu_vs_cpu", test_standard_map_gpu_vs_cpu);
    return t.summary();
}
