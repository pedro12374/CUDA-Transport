// Conservation checks: the standard sanity check for any integrator of an
// autonomous, structure-preserving system is that some analytically-known
// conserved quantity actually stays constant along the numerical trajectory.
//
//   - Horton, single-wave case (A2=A3=0): the system is then autonomous and
//     admits a co-moving-frame stream function psi(x,y) (see
//     maps/horton.h's horton_single_wave_stream_function and its derivation
//     comment) that's exactly conserved. Confirmed valid by direct
//     differentiation before implementing this test (see CLAUDE.md).
//   - Henon-Heiles: total energy H(x,y,px,py) is conserved for any lambda,
//     any initial condition -- the textbook Hamiltonian-integrator check.
//
// RK4 isn't symplectic, so exact conservation isn't expected -- but the
// drift should stay small and bounded (not grow secularly) over a
// reasonably long integration at a normal dt.
#include <cstdio>
#include <cmath>
#include "cuda_dynamics.h"
#include "horton.h"
#include "henon_heiles.h"
#include "common.h"

static bool test_horton_stream_function_conservation() {
    const int DIMS = 2;
    HortonSystem system;
    HortonSystemParams params = {
        .A1 = 1.0, .A2 = 0.0, .A3 = 0.0, // single-wave case: conservation should hold
        .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
        .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
        .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
    };
    params.v2 = 0.0;
    params.v3 = 0.0;

    double state[DIMS] = { 0.4, -0.7 };
    double psi0 = horton_single_wave_stream_function(state, params);

    // dt=0.005 (not 0.01): Horton's higher effective frequency scale
    // (kx1=6, ky1=3) needs a smaller dt for RK4's leading-order error term
    // to dominate at a given tolerance -- same finding as in
    // test_convergence.cu's horton_convergence.
    const double DT = 0.005;
    const int STEPS = 4000; // t_final = 20
    double max_drift = 0.0;
    for (int step = 0; step < STEPS; ++step) {
        cpu_rk4_step<DIMS, HortonSystem, HortonSystemParams>(state, step * DT, DT, system, params);
        double psi = horton_single_wave_stream_function(state, params);
        max_drift = std::max(max_drift, std::fabs(psi - psi0));
    }
    printf("         Horton (A2=A3=0): psi0=%.6f, max drift over %d steps (t=%.0f) = %.3e\n",
           psi0, STEPS, STEPS * DT, max_drift);
    return check(max_drift < 1e-5, "Horton single-wave stream function not conserved");
}

static bool test_henon_heiles_energy_conservation() {
    const int DIMS = 4;
    HenonHeilesSystem system;
    HenonHeilesParams params = { .lambda = 1.0 };

    double state[DIMS] = { 0.1, 0.15, 0.05, -0.05 }; // confined orbit (E well below E_c)
    double e0 = henon_heiles_energy(state, params);

    const double DT = 0.01;
    const int STEPS = 2000; // t_final = 20
    double max_drift = 0.0;
    for (int step = 0; step < STEPS; ++step) {
        cpu_rk4_step<DIMS, HenonHeilesSystem, HenonHeilesParams>(state, step * DT, DT, system, params);
        double e = henon_heiles_energy(state, params);
        max_drift = std::max(max_drift, std::fabs(e - e0));
    }
    printf("         Henon-Heiles: E0=%.6f, max drift over %d steps (t=%.0f) = %.3e\n",
           e0, STEPS, STEPS * DT, max_drift);
    return check(max_drift < 1e-6, "Henon-Heiles energy not conserved");
}

int main() {
    TestRunner t;
    t.run("horton_stream_function_conservation", test_horton_stream_function_conservation);
    t.run("henon_heiles_energy_conservation", test_henon_heiles_energy_conservation);
    return t.summary();
}
