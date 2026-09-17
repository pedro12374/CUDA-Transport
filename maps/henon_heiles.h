#pragma once
/**
 * @file henon_heiles.h
 * @brief The Henon-Heiles system -- a third worked example of the ODE
 * system interface (see horton.h's file-level docs for the full
 * four-piece contract). The main thing this example adds: `DIMS = 4`
 * (`x,y,px,py`), not 2 -- proof the interface, and the generic runner
 * (`run_ode_generic<DIMS,...>` in runner.cuh), are genuinely generic over
 * state dimensionality, not just over which 2D system is plugged in.
 *
 * The (generalized) Henon-Heiles system (Henon & Heiles, 1964):
 * @code
 *   H = 1/2(px^2 + py^2) + 1/2(x^2 + y^2) + lambda*(x^2*y - y^3/3)
 * @endcode
 * `lambda = 1` recovers the original Hamiltonian. A textbook example of a
 * bound Hamiltonian system with escape: for total energy E below the
 * critical value `E_c = 1/6`, every orbit is confined to a bounded
 * equipotential region; above `E_c`, three symmetric saddle points (at
 * `(0,1)`, `(-sqrt(3)/2,-1/2)`, `(sqrt(3)/2,-1/2)`, each at `V = 1/6`) open
 * onto unbounded escape channels, giving the classic fractal 3-basin
 * escape-time map this system is known for.
 */
#include <cmath>
#include "config.h"

template <typename SystemType>
struct SystemTraits;

/** @brief Henon-Heiles' one physical parameter: the cubic coupling strength (1.0 = the textbook system). */
struct HenonHeilesParams {
    double lambda;
};

/** @brief Hamilton's equations for the Henon-Heiles potential. State layout: `(x, y, px, py)`. */
struct HenonHeilesSystem {
    /**
     * @brief The RHS of `dstate/dt = f(state, t)` (see HortonSystem::operator()
     * in horton.h for the general contract this follows). Autonomous (no
     * explicit t-dependence), but the signature still takes `t` to match
     * the RK4 stepper's interface used throughout this library -- an
     * unused parameter costs nothing here.
     *
     * @tparam DIMS State dimensionality (4 for HenonHeilesSystem).
     * @param state Current state, `(x, y, px, py)`.
     * @param dstate_dt [out] Time derivative of `state`.
     * @param params Physical parameters.
     */
    template <int DIMS>
    __host__ __device__ void operator()(
        const double state[DIMS],
        double dstate_dt[DIMS],
        const HenonHeilesParams& params,
        double /*t*/) const
    {
        double x = state[0], y = state[1], px = state[2], py = state[3];
        dstate_dt[0] = px;
        dstate_dt[1] = py;
        dstate_dt[2] = -x - 2.0 * params.lambda * x * y;
        dstate_dt[3] = -y - params.lambda * (x * x - y * y);
    }

    /**
     * @brief Jacobian of operator()'s RHS, for the Lyapunov exponent solver
     * (see HortonSystem::jacobian() in horton.h for the general contract).
     * @tparam DIMS State dimensionality (4 for HenonHeilesSystem).
     * @param state Current state, `(x, y, px, py)`.
     * @param params Physical parameters.
     * @param J_out [out] The `DIMS x DIMS` Jacobian, flattened row-major.
     */
    template <int DIMS>
    __host__ __device__ void jacobian(
        const double state[DIMS],
        const HenonHeilesParams& params,
        double J_out[DIMS * DIMS],
        double /*t*/) const
    {
        double x = state[0], y = state[1];
        for (int i = 0; i < DIMS * DIMS; ++i) J_out[i] = 0.0;
        // state order (x, y, px, py); J_out[row*DIMS + col] = d(dstate_row/dt)/d(state_col)
        J_out[0 * DIMS + 2] = 1.0;                             // d(dx/dt)/dpx
        J_out[1 * DIMS + 3] = 1.0;                             // d(dy/dt)/dpy
        J_out[2 * DIMS + 0] = -1.0 - 2.0 * params.lambda * y;  // d(dpx/dt)/dx
        J_out[2 * DIMS + 1] = -2.0 * params.lambda * x;        // d(dpx/dt)/dy
        J_out[3 * DIMS + 0] = -2.0 * params.lambda * x;        // d(dpy/dt)/dx
        J_out[3 * DIMS + 1] = -1.0 + 2.0 * params.lambda * y;  // d(dpy/dt)/dy
    }
};

/**
 * @brief Total energy `H(x,y,px,py)` -- conserved along any trajectory of
 * HenonHeilesSystem. The standard correctness check for a Hamiltonian
 * integrator; see tests/test_conservation.cu.
 * @param state State, `(x, y, px, py)`.
 * @param params Physical parameters.
 * @return The Hamiltonian's value at `state`.
 */
__host__ __device__ inline double henon_heiles_energy(const double state[4], const HenonHeilesParams& params) {
    double x = state[0], y = state[1], px = state[2], py = state[3];
    return 0.5 * (px * px + py * py) + 0.5 * (x * x + y * y)
         + params.lambda * (x * x * y - (y * y * y) / 3.0);
}

/** @brief HenonHeilesSystem's trait specialization (see SystemTraits<HortonSystem> in horton.h for the general contract). */
template <>
struct SystemTraits<HenonHeilesSystem> {
    /** @brief Unbounded phase space: no periodic wrapping needed, so this is a no-op. */
    __host__ __device__ static void post_step_update(double /*state*/[4]) {}

    /**
     * @brief Escape test: triggers once clearly beyond the interaction
     * region (`r > 10`, well past the saddle points at `r = 1`).
     * @param state Current state, `(x, y, px, py)`.
     * @return `0` if not escaped. Otherwise, `1`/`2`/`3`, recording which
     * of the three 120-degree escape channels (centered at the saddle
     * points, 90/210/330 degrees) the trajectory left through.
     */
    __host__ __device__ static int check_escape(const double state[4]) {
        const double R_ESCAPE_SQ = 100.0; // r > 10
        double x = state[0], y = state[1];
        double r2 = x * x + y * y;
        if (r2 <= R_ESCAPE_SQ) return 0;

        double deg = atan2(y, x) * 180.0 / M_PI; // (-180, 180]
        double shifted = deg - 90.0;              // center channel 1 (90 deg) at 0
        while (shifted < -180.0) shifted += 360.0;
        while (shifted >= 180.0) shifted -= 360.0;
        if (shifted >= -60.0 && shifted < 60.0) return 1;  // near 90 deg
        if (shifted >= 60.0 && shifted < 180.0) return 2;  // near 210 deg
        return 3;                                           // near 330 deg (-30 deg)
    }
};

/** @brief Builds HenonHeilesParams from a resolved config (see config.h). */
template <>
inline HenonHeilesParams load_params_for<HenonHeilesParams>(const Config& cfg) {
    HenonHeilesParams p;
    p.lambda = cfg.get_double("lambda", 1.0);
    return p;
}
