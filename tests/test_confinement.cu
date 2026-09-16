// Regular orbits in a near-integrable/sub-critical regime should stay
// confined (never trigger the escape criterion), however long they run:
//   - Standard Map well below the chaos threshold K_c ~ 0.9716.
//   - Henon-Heiles well below the critical energy E_c = 1/6 ~ 0.1667.
#include <cstdio>
#include <cmath>
#include <vector>
#include "cuda_dynamics.h"
#include "standard_map.h"
#include "henon_heiles.h"
#include "common.h"

static bool test_standard_map_low_k_confined() {
    const int DIMS = 2;
    StandardMap map;
    StandardMapParams params = { .K = 0.3 }; // well below K_c ~ 0.9716

    // Sub-critical K doesn't mean *every* orbit is confined -- only ones
    // well inside a stable island are; the escape boundary is |p| > pi, and
    // orbits starting in the local stochastic layer near a separatrix can
    // still wander within a bounded-but-wide band and cross it even at low
    // K. p0 restricted to a small range around p=0 (deep inside the main
    // resonance island) is what's actually guaranteed confined; verified
    // empirically to hold up to p0 in [-0.5,0.5] at this K.
    GridSetup grid(DIMS, {12, 12}, { -0.5, 0.0 }, { 0.5, 2.0 * M_PI });

    std::vector<double> h_times(grid.num_particles), h_basins(grid.num_particles);
    const int ITERATIONS = 100000;
    calculate_escape_time<DIMS, StandardMap, StandardMapParams>(
        map, params, grid.h_initial_conditions.data(), grid.num_particles,
        ITERATIONS, h_times.data(), h_basins.data());

    long long escaped = 0;
    for (long long i = 0; i < grid.num_particles; ++i) {
        if (h_times[i] != -1.0) ++escaped;
    }
    printf("         Standard Map K=%.2f: %lld/%lld particles escaped over %d iterations\n",
           params.K, escaped, grid.num_particles, ITERATIONS);
    return check(escaped == 0, "Standard Map: regular orbits escaped at low K");
}

static bool test_henon_heiles_low_energy_confined() {
    const int DIMS = 4;
    HenonHeilesSystem system;
    HenonHeilesParams params = { .lambda = 1.0 };

    // Released from rest (px0=py0=0), so the energy at each grid point is
    // just V(x0,y0) = 1/2(x0^2+y0^2) + x0^2*y0 - y0^3/3. Keep the (x0,y0)
    // box small enough that V stays comfortably below E_c=1/6 everywhere
    // in it (checked below, not just assumed).
    GridSetup grid(DIMS, {10, 10, 1, 1}, { -0.3, -0.3, 0.0, 0.0 }, { 0.3, 0.3, 0.0, 0.0 });

    double max_v = 0.0;
    for (long long i = 0; i < grid.num_particles; ++i) {
        double state[DIMS];
        for (int j = 0; j < DIMS; ++j) state[j] = grid.h_initial_conditions[i * DIMS + j];
        max_v = std::max(max_v, henon_heiles_energy(state, params)); // px=py=0, so H==V here
    }
    const double E_CRIT = 1.0 / 6.0;
    if (!check(max_v < E_CRIT, "Henon-Heiles confinement test grid isn't actually sub-critical")) {
        printf("         max V(x0,y0) = %.4f, E_c = %.4f\n", max_v, E_CRIT);
        return false;
    }

    std::vector<double> h_times(grid.num_particles), h_basins(grid.num_particles);
    const double DT = 0.01;
    const int TOTAL_STEPS = 500000; // T_final = 5000, long enough to be a meaningful confinement check
    calculate_ode_escape<DIMS, HenonHeilesSystem, HenonHeilesParams>(
        system, params, grid.h_initial_conditions.data(), grid.num_particles,
        TOTAL_STEPS, DT, h_times.data(), h_basins.data());

    long long escaped = 0;
    for (long long i = 0; i < grid.num_particles; ++i) {
        if (h_times[i] != -1.0) ++escaped;
    }
    printf("         Henon-Heiles (max V=%.4f < E_c=%.4f): %lld/%lld particles escaped over t=%.0f\n",
           max_v, E_CRIT, escaped, grid.num_particles, TOTAL_STEPS * DT);
    return check(escaped == 0, "Henon-Heiles: sub-critical-energy orbits escaped");
}

int main() {
    TestRunner t;
    t.run("standard_map_low_k_confined", test_standard_map_low_k_confined);
    t.run("henon_heiles_low_energy_confined", test_henon_heiles_low_energy_confined);
    return t.summary();
}
