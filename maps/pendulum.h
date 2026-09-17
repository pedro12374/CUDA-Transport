#pragma once
/**
 * @file pendulum.h
 * @brief The driven, damped pendulum -- a fourth worked example of the ODE
 * system interface, built from scratch in TUTORIAL.md. See horton.h's
 * file-level docs for the full four-piece contract this follows.
 *
 * Equation of motion (theta = angle from the downward vertical, p = dtheta/dt):
 * @code
 *   d(theta)/dt = p
 *   dp/dt = -gamma*p - sin(theta) + A*cos(omega_drive*t)
 * @endcode
 * A damped pendulum (damping gamma) driven by a periodic torque of
 * amplitude A and frequency omega_drive -- the standard "chaotically
 * driven pendulum" model (see e.g. Strogatz, *Nonlinear Dynamics and
 * Chaos*). theta is left unwrapped (see SystemTraits<PendulumSystem>
 * below): a trajectory that keeps rotating over the top, rather than
 * oscillating back and forth, is what this system calls "escaped."
 */
#include <cmath>
#include "config.h"

template <typename SystemType>
struct SystemTraits;

/** @brief Physical parameters for PendulumSystem: damping, drive amplitude, drive frequency. */
struct PendulumParams {
    double gamma;       ///< Damping coefficient.
    double A;            ///< Driving torque amplitude.
    double omega_drive;  ///< Driving frequency.
};

/** @brief The driven, damped pendulum. State layout: `(theta, p)`. */
struct PendulumSystem {
    /**
     * @brief The RHS of `dstate/dt = f(state, t)` (see HortonSystem::operator()
     * in horton.h for the general contract).
     * @tparam DIMS State dimensionality (2 for PendulumSystem).
     * @param state Current state, `(theta, p)`.
     * @param dstate_dt [out] Time derivative of `state`.
     * @param params Physical parameters.
     * @param t Current simulation time (needed here: the driving torque depends on it).
     */
    template <int DIMS>
    __host__ __device__ void operator()(
        const double state[DIMS],
        double dstate_dt[DIMS],
        const PendulumParams& params,
        double t) const
    {
        double theta = state[0];
        double p = state[1];
        dstate_dt[0] = p;
        dstate_dt[1] = -params.gamma * p - sin(theta) + params.A * cos(params.omega_drive * t);
    }

    /**
     * @brief Jacobian of operator()'s RHS, for the Lyapunov exponent solver
     * (see HortonSystem::jacobian() in horton.h for the general contract).
     * @tparam DIMS State dimensionality (2 for PendulumSystem).
     * @param state Current state, `(theta, p)`.
     * @param params Physical parameters.
     * @param J_out [out] The `DIMS x DIMS` Jacobian, flattened row-major.
     * @param t Current simulation time (unused: the driving torque doesn't depend on state).
     */
    template <int DIMS>
    __host__ __device__ void jacobian(
        const double state[DIMS],
        const PendulumParams& params,
        double J_out[DIMS * DIMS],
        double /*t*/) const
    {
        double theta = state[0];
        J_out[0] = 0.0;             // d(dtheta/dt)/dtheta
        J_out[1] = 1.0;             // d(dtheta/dt)/dp
        J_out[2] = -cos(theta);     // d(dp/dt)/dtheta
        J_out[3] = -params.gamma;   // d(dp/dt)/dp
    }
};

/** @brief PendulumSystem's trait specialization (see SystemTraits<HortonSystem> in horton.h for the general contract). */
template <>
struct SystemTraits<PendulumSystem> {
    /**
     * @brief theta is deliberately left unwrapped -- the same pattern
     * maps/standard_map.h uses for its momentum coordinate -- so
     * check_escape() below can use its accumulated value directly as a
     * net-rotation count. A no-op.
     */
    __host__ __device__ static void post_step_update(double /*state*/[2]) {}

    /**
     * @brief Escape test: fires once the pendulum has completed enough net
     * rotations to be considered "escaped" into continuous rotation,
     * rather than trapped oscillating near the bottom.
     * @param state Current state, `(theta, p)`.
     * @return `0` if not escaped. Otherwise `1` (rotating counter/anticlockwise
     * per the sign of `theta`, i.e. `theta > N_ROTATIONS*2*pi`) or `-1`
     * (`theta < -N_ROTATIONS*2*pi`).
     */
    __host__ __device__ static int check_escape(const double state[2]) {
        const double N_ROTATIONS = 3.0;
        const double THRESHOLD = N_ROTATIONS * 2.0 * M_PI;
        double theta = state[0];
        if (theta > THRESHOLD) return 1;
        if (theta < -THRESHOLD) return -1;
        return 0;
    }
};

/** @brief Builds PendulumParams from a resolved config (see config.h). */
template <>
inline PendulumParams load_params_for<PendulumParams>(const Config& cfg) {
    PendulumParams p;
    p.gamma = cfg.get_double("gamma", 0.5);
    p.A = cfg.get_double("A", 1.5);
    p.omega_drive = cfg.get_double("omega_drive", 0.666667);
    return p;
}
