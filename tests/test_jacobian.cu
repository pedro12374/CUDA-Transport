// Validates every system's jacobian() against a finite-difference check of
// the function it's supposed to differentiate (operator()). This class of
// bug (Horton's Jacobian used a phase constant that didn't match its own
// RHS) previously went undetected because no test exercised the
// Lyapunov/jacobian path at all -- this file closes that gap for every
// system in the library.
#include <cstdio>
#include <cmath>
#include <string>
#include <algorithm>
#include "horton.h"
#include "henon_heiles.h"
#include "pendulum.h"
#include "standard_map.h"
#include "common.h"

// ODE systems: jacobian() differentiates operator()'s RHS w.r.t. state.
template <int DIMS, typename SystemType, typename ParamsType>
static bool check_ode_jacobian(const char* name, const SystemType& sys, const ParamsType& params,
                                const double state[DIMS], double t, double tol) {
    double J[DIMS * DIMS];
    sys.template jacobian<DIMS>(state, params, J, t);

    double Jfd[DIMS * DIMS];
    const double h = 1e-6;
    for (int j = 0; j < DIMS; ++j) {
        double sp[DIMS], sm[DIMS], fp[DIMS], fm[DIMS];
        for (int i = 0; i < DIMS; ++i) { sp[i] = state[i]; sm[i] = state[i]; }
        sp[j] += h; sm[j] -= h;
        sys.template operator()<DIMS>(sp, fp, params, t);
        sys.template operator()<DIMS>(sm, fm, params, t);
        for (int i = 0; i < DIMS; ++i) Jfd[i * DIMS + j] = (fp[i] - fm[i]) / (2 * h);
    }

    double max_diff = 0.0;
    for (int i = 0; i < DIMS * DIMS; ++i) max_diff = std::max(max_diff, std::fabs(J[i] - Jfd[i]));
    printf("         %s: max |analytic - finite-diff| = %.3e\n", name, max_diff);
    return check(max_diff < tol, std::string(name) + " jacobian doesn't match finite-difference RHS derivative");
}

// Discrete maps: jacobian() differentiates operator()'s state update
// (state_new = g(state_old)) w.r.t. the input state.
template <int DIMS, typename MapType, typename ParamsType>
static bool check_map_jacobian(const char* name, const MapType& map, const ParamsType& params,
                                const double state[DIMS], double tol) {
    double J[DIMS * DIMS];
    map.template jacobian<DIMS>(state, params, J);

    double Jfd[DIMS * DIMS];
    const double h = 1e-6;
    for (int j = 0; j < DIMS; ++j) {
        double sp[DIMS], sm[DIMS];
        for (int i = 0; i < DIMS; ++i) { sp[i] = state[i]; sm[i] = state[i]; }
        sp[j] += h; sm[j] -= h;
        map.template operator()<DIMS>(sp, nullptr, params);
        map.template operator()<DIMS>(sm, nullptr, params);
        for (int i = 0; i < DIMS; ++i) Jfd[i * DIMS + j] = (sp[i] - sm[i]) / (2 * h);
    }

    double max_diff = 0.0;
    for (int i = 0; i < DIMS * DIMS; ++i) max_diff = std::max(max_diff, std::fabs(J[i] - Jfd[i]));
    printf("         %s: max |analytic - finite-diff| = %.3e\n", name, max_diff);
    return check(max_diff < tol, std::string(name) + " jacobian doesn't match finite-difference map derivative");
}

static bool test_horton_jacobian() {
    HortonSystem sys;
    HortonSystemParams params = {
        .A1 = 1.0, .A2 = 0.5, .A3 = 0.3,
        .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
        .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
        .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
    };
    params.v2 = std::fabs(params.w2 / params.ky2 - params.w1 / params.ky1);
    params.v3 = std::fabs(params.w3 / params.ky3 - params.w1 / params.ky1);
    double state[2] = { 0.37, -1.21 };
    // A2, A3 both nonzero: this is exactly the case that was broken.
    return check_ode_jacobian<2>("Horton (A2,A3 != 0)", sys, params, state, 3.7, 1e-6);
}

static bool test_henon_heiles_jacobian() {
    HenonHeilesSystem sys;
    HenonHeilesParams params = { .lambda = 1.0 };
    double state[4] = { 0.1, 0.15, 0.05, -0.05 };
    return check_ode_jacobian<4>("Henon-Heiles", sys, params, state, 0.0, 1e-6);
}

static bool test_pendulum_jacobian() {
    PendulumSystem sys;
    PendulumParams params = { .gamma = 0.5, .A = 1.5, .omega_drive = 0.666667 };
    double state[2] = { 0.3, -0.2 };
    return check_ode_jacobian<2>("Pendulum", sys, params, state, 1.5, 1e-6);
}

static bool test_standard_map_jacobian() {
    StandardMap map;
    StandardMapParams params = { .K = 1.5 };
    double state[2] = { 0.2, 1.0 }; // (p, theta), theta away from the fmod wrap boundary
    return check_map_jacobian<2>("Standard Map", map, params, state, 1e-6);
}

int main() {
    TestRunner t;
    t.run("horton_jacobian", test_horton_jacobian);
    t.run("henon_heiles_jacobian", test_henon_heiles_jacobian);
    t.run("pendulum_jacobian", test_pendulum_jacobian);
    t.run("standard_map_jacobian", test_standard_map_jacobian);
    return t.summary();
}
