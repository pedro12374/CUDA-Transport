#pragma once // File: tests/common.h
//
// Minimal, dependency-free test helpers shared by tests/test_*.cu:
//   - a tiny pass/fail test framework (no external test library needed)
//   - an independently-written CPU RK4 stepper, for GPU-vs-CPU and
//     convergence-order checks against the library's actual RK4
//     kernels/rk4_step_t

#include <cstdio>
#include <cmath>
#include <string>
#include <functional>
#include <vector>

// =============================================================================
// == Tiny test framework
// =============================================================================

struct TestRunner {
    int passed = 0;
    int failed = 0;

    void run(const std::string& name, const std::function<bool()>& test) {
        printf("[ RUN  ] %s\n", name.c_str());
        bool ok = false;
        try {
            ok = test();
        } catch (const std::exception& e) {
            printf("[ERROR ] %s: uncaught exception: %s\n", name.c_str(), e.what());
            ok = false;
        }
        if (ok) {
            printf("[  OK  ] %s\n", name.c_str());
            ++passed;
        } else {
            printf("[ FAIL ] %s\n", name.c_str());
            ++failed;
        }
    }

    // Prints a summary and returns a process exit code (0 = all passed).
    int summary() const {
        printf("\n%d passed, %d failed\n", passed, failed);
        return failed == 0 ? 0 : 1;
    }
};

// Prints a diagnostic line and returns false when `cond` fails -- meant to
// be used as `if (!check(...)) return false;` inside a test lambda.
inline bool check(bool cond, const std::string& msg) {
    if (!cond) {
        printf("         check failed: %s\n", msg.c_str());
    }
    return cond;
}

inline bool check_near(double actual, double expected, double tol, const std::string& what) {
    double diff = std::fabs(actual - expected);
    if (diff > tol) {
        printf("         %s: expected %.6g, got %.6g (diff %.3e > tol %.3e)\n",
               what.c_str(), expected, actual, diff, tol);
        return false;
    }
    return true;
}

// =============================================================================
// == Independent CPU RK4 stepper
// =============================================================================
//
// Deliberately re-typed rather than reusing rk4_step_t from
// cuda_dynamics.h, so it's a genuine independent check of the GPU
// integrator's stage-time logic, not just the same code re-run on the host.

template <int DIMS, typename SystemType, typename ParamsType>
inline void cpu_rk4_step(double state[DIMS], double t, double dt,
                          const SystemType& system, const ParamsType& params) {
    double k1[DIMS], k2[DIMS], k3[DIMS], k4[DIMS], tmp[DIMS];

    system.template operator()<DIMS>(state, k1, params, t);
    for (int i = 0; i < DIMS; ++i) tmp[i] = state[i] + 0.5 * dt * k1[i];

    system.template operator()<DIMS>(tmp, k2, params, t + 0.5 * dt);
    for (int i = 0; i < DIMS; ++i) tmp[i] = state[i] + 0.5 * dt * k2[i];

    system.template operator()<DIMS>(tmp, k3, params, t + 0.5 * dt);
    for (int i = 0; i < DIMS; ++i) tmp[i] = state[i] + dt * k3[i];

    system.template operator()<DIMS>(tmp, k4, params, t + dt);

    for (int i = 0; i < DIMS; ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

// Integrates from t=0 to t=t_final with a fixed step dt (t_final assumed to
// be an exact multiple of dt), using cpu_rk4_step above.
template <int DIMS, typename SystemType, typename ParamsType>
inline void cpu_integrate(double state[DIMS], double t_final, double dt,
                           const SystemType& system, const ParamsType& params) {
    int num_steps = static_cast<int>(std::round(t_final / dt));
    for (int step = 0; step < num_steps; ++step) {
        cpu_rk4_step<DIMS, SystemType, ParamsType>(state, step * dt, dt, system, params);
    }
}
