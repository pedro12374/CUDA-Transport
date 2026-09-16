// Convergence as dt decreases: RK4 is a 4th-order method, so halving dt
// should shrink the global error by ~2^4 = 16x. Checked entirely on the
// CPU reference stepper (tests/common.h's cpu_rk4_step, the same one
// validated against the real GPU kernels in test_gpu_vs_cpu.cu) by
// integrating the same trajectory at dt, dt/2, and dt/4, using the dt/4
// result as a high-accuracy reference:
//   error(dt)    = |sol(dt)    - sol(dt/4)|
//   error(dt/2)  = |sol(dt/2)  - sol(dt/4)|
//   ratio = error(dt) / error(dt/2), expected ~16 for a 4th-order method
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "cuda_dynamics.h"
#include "horton.h"
#include "henon_heiles.h"
#include "common.h"

template <int DIMS, typename SystemType, typename ParamsType>
static double max_component_diff(const double a[DIMS], const double b[DIMS]) {
    double d = 0.0;
    for (int i = 0; i < DIMS; ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

static bool test_horton_convergence() {
    const int DIMS = 2;
    const double T_FINAL = 2.0;
    // 0.005 (not larger) so the leading-order O(dt^4) term actually
    // dominates -- at dt=0.02 the ratio is still ~7.8, not yet ~16;
    // verified empirically it settles in around dt=0.005.
    const double DT = 0.005; // T_FINAL/DT, /2, /4 all exact integers

    HortonSystem system;
    HortonSystemParams params = {
        .A1 = 1.0, .A2 = 0.5, .A3 = 0.3,
        .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
        .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
        .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
    };
    params.v2 = std::fabs(params.w2 / params.ky2 - params.w1 / params.ky1);
    params.v3 = std::fabs(params.w3 / params.ky3 - params.w1 / params.ky1);

    double ic[DIMS] = { 0.3, -1.1 };

    double s_dt[DIMS], s_dt2[DIMS], s_dt4[DIMS];
    std::copy(ic, ic + DIMS, s_dt);
    std::copy(ic, ic + DIMS, s_dt2);
    std::copy(ic, ic + DIMS, s_dt4);

    cpu_integrate<DIMS>(s_dt, T_FINAL, DT, system, params);
    cpu_integrate<DIMS>(s_dt2, T_FINAL, DT / 2.0, system, params);
    cpu_integrate<DIMS>(s_dt4, T_FINAL, DT / 4.0, system, params);

    double err_dt = max_component_diff<DIMS, HortonSystem, HortonSystemParams>(s_dt, s_dt4);
    double err_dt2 = max_component_diff<DIMS, HortonSystem, HortonSystemParams>(s_dt2, s_dt4);
    double ratio = err_dt / err_dt2;

    printf("         Horton: err(dt)=%.3e err(dt/2)=%.3e ratio=%.2f (expect ~16 for 4th order)\n",
           err_dt, err_dt2, ratio);
    return check(ratio > 8.0 && ratio < 32.0, "Horton RK4 convergence ratio not ~16");
}

static bool test_henon_heiles_convergence() {
    const int DIMS = 4;
    const double T_FINAL = 2.0;
    const double DT = 0.02;

    HenonHeilesSystem system;
    HenonHeilesParams params = { .lambda = 1.0 };

    // Low energy (well below critical E_c=1/6), released from rest --
    // regular, non-chaotic orbit, appropriate for a clean convergence check.
    double ic[DIMS] = { 0.1, 0.15, 0.0, 0.0 };

    double s_dt[DIMS], s_dt2[DIMS], s_dt4[DIMS];
    std::copy(ic, ic + DIMS, s_dt);
    std::copy(ic, ic + DIMS, s_dt2);
    std::copy(ic, ic + DIMS, s_dt4);

    cpu_integrate<DIMS>(s_dt, T_FINAL, DT, system, params);
    cpu_integrate<DIMS>(s_dt2, T_FINAL, DT / 2.0, system, params);
    cpu_integrate<DIMS>(s_dt4, T_FINAL, DT / 4.0, system, params);

    double err_dt = max_component_diff<DIMS, HenonHeilesSystem, HenonHeilesParams>(s_dt, s_dt4);
    double err_dt2 = max_component_diff<DIMS, HenonHeilesSystem, HenonHeilesParams>(s_dt2, s_dt4);
    double ratio = err_dt / err_dt2;

    printf("         Henon-Heiles: err(dt)=%.3e err(dt/2)=%.3e ratio=%.2f (expect ~16 for 4th order)\n",
           err_dt, err_dt2, ratio);
    return check(ratio > 8.0 && ratio < 32.0, "Henon-Heiles RK4 convergence ratio not ~16");
}

int main() {
    TestRunner t;
    t.run("horton_convergence", test_horton_convergence);
    t.run("henon_heiles_convergence", test_henon_heiles_convergence);
    return t.summary();
}
