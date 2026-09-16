#pragma once
/**
 * @file horton.h
 * @brief The Horton (drift-wave / three-wave) system -- a worked example
 * of the complete **ODE system interface** a new continuous-time system
 * must implement to plug into the generic runner (see run_ode_generic() in
 * runner.cuh, and TUTORIAL.md for a from-scratch walkthrough building a
 * second one).
 *
 * An ODE system needs exactly four pieces, all present in this file:
 *   1. A `*Params` struct holding its physical parameters (here,
 *      HortonSystemParams).
 *   2. A functor with a templated `operator()(state, dstate_dt, params, t)`
 *      -- the right-hand side of `dstate/dt = f(state, t)` -- and a
 *      templated `jacobian(state, params, J_out, t)` for the Lyapunov
 *      solver (see HortonSystem below).
 *   3. A `SystemTraits<YourSystem>` specialization providing
 *      `post_step_update(state)` (periodic wrapping, or a no-op if the
 *      phase space is unbounded) and `check_escape(state)` (see
 *      SystemTraits<HortonSystem> below).
 *   4. A `load_params_for<YourParams>` specialization (config.h) building
 *      the params struct from a config file.
 */
#include <cmath>
#include "config.h"

template <typename SystemType>
struct SystemTraits;

/**
 * @brief Physical parameters for HortonSystem: amplitude, wavevector
 * (kx,ky), and frequency (w) for each of the three waves, plus the
 * co-moving-frame velocities v2,v3 derived from them (see
 * load_params_for<HortonSystemParams> below).
 */
struct HortonSystemParams {
    /** Wave amplitudes. */
    double A1, A2, A3;

    /** Wave 1: wavevector components and frequency. */
    double kx1, ky1, w1;
    /** Wave 2: wavevector components and frequency. */
    double kx2, ky2, w2;
    /** Wave 3: wavevector components and frequency. */
    double kx3, ky3, w3;

    /** Phase velocities of waves 2 and 3 relative to wave 1 (derived, not independent). */
    double v2, v3;
};

/**
 * @brief The three-wave drift system: a 2D (x,y) flow forced by one
 * stationary wave (amplitude A1) and two waves (A2,A3) advecting past it
 * at velocities v2,v3, hence explicitly time-dependent whenever A2 or A3
 * is nonzero. State layout: `(x, y)`.
 */
struct HortonSystem {
    /**
     * @brief The RHS of `dstate/dt = f(state, t)`: `dstate_dt[i] = d(state[i])/dt`.
     *
     * Called at every RK4 stage (see rk4_step_t() in cuda_dynamics.h) with
     * the appropriate intermediate state and time -- must be a pure
     * function of its arguments (no hidden state), and callable from both
     * host and device code, which is why it's marked `__host__ __device__`
     * and why `state`/`dstate_dt` are fixed-size C arrays rather than
     * something like `std::vector`.
     *
     * @tparam DIMS State dimensionality (2 for HortonSystem).
     * @param state Current state, `(x, y)`.
     * @param dstate_dt [out] Time derivative of `state`.
     * @param params Physical parameters.
     * @param t Current simulation time (this system needs it whenever A2/A3 != 0).
     */
    template <int DIMS>
    __host__ __device__ void operator()(
        const double state[DIMS],
        double dstate_dt[DIMS],
        const HortonSystemParams& params,
        double t // Current time
    ) const {
        double x = state[0];
        double y = state[1];

        // Term 1
        double term1_x = params.A1 * params.ky1 * sin(params.kx1 * x) * sin(params.ky1 * y);
        double term1_y = params.A1 * params.kx1 * cos(params.kx1 * x) * cos(params.ky1 * y);

        // Term 2
        double arg2_y = params.ky2 * (y - params.v2 * t);
        double term2_x = params.A2 * params.ky2 * sin(params.kx2 * x + 0.5) * sin(arg2_y);
        double term2_y = params.A2 * params.kx2 * cos(params.kx2 * x + 0.5) * cos(arg2_y);

        // Term 3
        double arg3_y = params.ky3 * (y - params.v3 * t);
        double term3_x = params.A3 * params.ky3 * sin(params.kx3 * x + 0.5) * sin(arg3_y);
        double term3_y = params.A3 * params.kx3 * cos(params.kx3 * x + 0.5) * cos(arg3_y);

        dstate_dt[0] = term1_x + term2_x + term3_x; // dxdt
        dstate_dt[1] = term1_y + term2_y + term3_y; // dydt

        

    }

    /**
     * @brief Jacobian of operator()'s RHS: `J_out[i*DIMS+j] = d(dstate_dt[i])/d(state[j])`,
     * row-major. Used only by the Lyapunov exponent solver
     * (calculate_ode_lyapunov_exponent() in ode_lyapunov.cuh) to evolve a
     * tangent vector; systems that never need a Lyapunov exponent can skip
     * this (it's a template member, so it's only instantiated -- and thus
     * only needs to compile -- if actually called).
     *
     * @tparam DIMS State dimensionality (2 for HortonSystem).
     * @param state Current state, `(x, y)`.
     * @param params Physical parameters.
     * @param J_out [out] The `DIMS x DIMS` Jacobian, flattened row-major.
     * @param t Current simulation time.
     */
    template <int DIMS>
    __host__ __device__ void jacobian(
        const double state[DIMS],
        const HortonSystemParams& params,
        double J_out[DIMS * DIMS],
        double t
    ) const {
        double x = state[0];
        double y = state[1];

        // Pre-calculate common terms
        double cos_kx1_x = cos(params.kx1 * x);
        double sin_kx1_x = sin(params.kx1 * x);
        double cos_ky1_y = cos(params.ky1 * y);
        double sin_ky1_y = sin(params.ky1 * y);
        
        // Phase offset must match operator()'s sin(kx2*x + 0.5)/cos(kx2*x + 0.5)
        // (and the wave-3 equivalent) exactly -- this used to be M_PI, which
        // silently differentiated a different function (cos(theta+pi) =
        // -cos(theta) != cos(theta+0.5) in general), giving a wrong Jacobian
        // -- and hence a wrong Lyapunov exponent -- whenever A2 or A3 != 0.
        double cos_kx2_x_phase = cos(params.kx2 * x + 0.5);
        double sin_kx2_x_phase = sin(params.kx2 * x + 0.5);
        double arg2_y = params.ky2 * (y - params.v2 * t);
        double cos_arg2_y = cos(arg2_y);
        double sin_arg2_y = sin(arg2_y);

        double cos_kx3_x_phase = cos(params.kx3 * x + 0.5);
        double sin_kx3_x_phase = sin(params.kx3 * x + 0.5);
        double arg3_y = params.ky3 * (y - params.v3 * t);
        double cos_arg3_y = cos(arg3_y);
        double sin_arg3_y = sin(arg3_y);

        // df/dx (J_out[0])
        J_out[0] = params.A1 * params.ky1 * params.kx1 * cos_kx1_x * sin_ky1_y +
                   params.A2 * params.ky2 * params.kx2 * cos_kx2_x_phase * sin_arg2_y +
                   params.A3 * params.ky3 * params.kx3 * cos_kx3_x_phase * sin_arg3_y;

        // df/dy (J_out[1])
        J_out[1] = params.A1 * params.ky1 * params.ky1 * sin_kx1_x * cos_ky1_y +
                   params.A2 * params.ky2 * params.ky2 * sin_kx2_x_phase * cos_arg2_y +
                   params.A3 * params.ky3 * params.ky3 * sin_kx3_x_phase * cos_arg3_y;

        // dg/dx (J_out[2])
        J_out[2] = -params.A1 * params.kx1 * params.kx1 * sin_kx1_x * cos_ky1_y -
                    params.A2 * params.kx2 * params.kx2 * sin_kx2_x_phase * cos_arg2_y -
                    params.A3 * params.kx3 * params.kx3 * sin_kx3_x_phase * cos_arg3_y;
                    
        // dg/dy (J_out[3])
        J_out[3] = -params.A1 * params.kx1 * params.ky1 * cos_kx1_x * sin_ky1_y -
                    params.A2 * params.kx2 * params.ky2 * cos_kx2_x_phase * sin_arg2_y -
                    params.A3 * params.kx3 * params.ky3 * cos_kx3_x_phase * sin_arg3_y;
    }
};

/**
 * @brief HortonSystem's trait specialization: the periodic-boundary and
 * escape-criterion half of the ODE system interface (see horton.h's
 * file-level docs for the full four-piece contract). Every `SystemTraits<T>`
 * specialization must provide exactly these two static members.
 */
template<>
struct SystemTraits<HortonSystem> {

    /**
     * @brief Called once per integration step, after the RK4 update, to
     * enforce boundary conditions on the raw state (e.g. wrap an angle
     * into its periodic domain). For a system with an unbounded phase
     * space, this can just be a no-op (see e.g.
     * SystemTraits<HenonHeilesSystem>).
     *
     * HortonSystem's y-coordinate is periodic with period 4*pi (domain
     * `[-2*pi, 2*pi]`); x is left unwrapped so check_escape() below can use
     * it directly as an escape distance. Wrapping here (rather than never)
     * matters for the MSD solver in particular: it tracks *unwrapped*
     * displacement separately (ode_msd.cuh) precisely so wrapping the
     * dynamics' own coordinate doesn't corrupt the displacement measurement.
     *
     * @param state [in,out] State to wrap in place, `(x, y)`.
     */
    __host__ __device__ static void post_step_update(double state[2]) {
    // Defines the periodic domain [-L/2, L/2]
    const double PERIOD = 4.0 * M_PI; // The total length of the interval (2pi - (-2pi))
    const double MIN_BOUND = -2.0 * M_PI;

    // Shift the value so the range starts at 0
    double temp = state[1] - MIN_BOUND;

    // Apply fmod
    temp = fmod(temp, PERIOD);

    // If the result is negative, add the period to wrap it around correctly
    if (temp < 0.0) {
        temp += PERIOD;
    }

    // Shift the value back to the original range
    state[1] = temp + MIN_BOUND;
}


    /**
     * @brief Called after every post_step_update() to test whether a
     * trajectory has escaped, and if so, which basin it left through.
     *
     * @param state Current (already-wrapped) state, `(x, y)`.
     * @return `0` if not escaped. Otherwise, a nonzero basin ID: `1` for
     * `x > pi` ("escapes up"), `-1` for `x < -pi` ("escapes down"). The
     * specific IDs are this system's choice -- the only rule the rest of
     * the library relies on is that `0` means "still confined."
     */
    __host__ __device__ static int check_escape(const double state[2]) {
        // Example: Define two distinct, non-symmetric escape regions

        // Basin 1: Escapes "up"
        if (state[0] > M_PI) {
            return 1;
        }

        // Basin 2: Escapes "down"
        if (state[0] < -M_PI) {
            return -1;
        }

        // Add other conditions here, e.g., escape left/right
        // if (state[0] > M_PI) { return 2; }

        // If no condition is met, it has not escaped.
        return 0;
    }
};

/**
 * @brief Co-moving-frame stream function for the SINGLE-WAVE case
 * (`A2 = A3 = 0`), where the system is autonomous and this quantity is
 * exactly conserved along any trajectory (see tests/test_conservation.cu):
 * @code
 *   dx/dt = A1*ky1*sin(kx1*x)*sin(ky1*y) = d(psi)/dy
 *   dy/dt = A1*kx1*cos(kx1*x)*cos(ky1*y) = -d(psi)/dx
 * @endcode
 * with `psi(x,y) = -A1*sin(kx1*x)*cos(ky1*y)` (verified by direct
 * differentiation).
 *
 * @warning NOT conserved when A2 or A3 != 0: those terms make the system
 * explicitly time-dependent (through v2*t, v3*t), so there's no autonomous
 * stream function for the full three-wave system.
 *
 * @param state State, `(x, y)`.
 * @param params Physical parameters (only A1, kx1, ky1 matter here).
 * @return The stream function's value at `state`.
 */
__host__ __device__ inline double horton_single_wave_stream_function(
    const double state[2], const HortonSystemParams& params) {
    return -params.A1 * sin(params.kx1 * state[0]) * cos(params.ky1 * state[1]);
}

/**
 * @brief Builds HortonSystemParams from a resolved config (see config.h);
 * the last piece of the ODE system interface described in this file's
 * top-level docs. Defaults match the values previously hardcoded in
 * main_horton_escape.cu/main_horton_msd.cu, so an incomplete config still
 * runs something sensible.
 */
template <>
inline HortonSystemParams load_params_for<HortonSystemParams>(const Config& cfg) {
    HortonSystemParams p;
    p.A1 = cfg.get_double("A1", 1.0);
    p.A2 = cfg.get_double("A2", 0.0);
    p.A3 = cfg.get_double("A3", 0.0);
    p.kx1 = cfg.get_double("kx1", 6.0);
    p.ky1 = cfg.get_double("ky1", 3.0);
    p.w1  = cfg.get_double("w1", 0.476);
    p.kx2 = cfg.get_double("kx2", -3.5);
    p.ky2 = cfg.get_double("ky2", -1.5);
    p.w2  = cfg.get_double("w2", 0.476);
    p.kx3 = cfg.get_double("kx3", -2.5);
    p.ky3 = cfg.get_double("ky3", -1.5);
    p.w3  = cfg.get_double("w3", 0.476);
    p.v2 = std::fabs(p.w2 / p.ky2 - p.w1 / p.ky1);
    p.v3 = std::fabs(p.w3 / p.ky3 - p.w1 / p.ky1);
    return p;
}