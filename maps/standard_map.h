#pragma once
/**
 * @file standard_map.h
 * @brief The (transport-modified) Standard Map -- a worked example of the
 * **discrete-map system interface**, the counterpart to horton.h's ODE
 * interface for systems advanced by `run_map_generic()` (runner.cuh)
 * instead of `run_ode_generic()`.
 *
 * A map system needs the same four kinds of pieces as an ODE system (see
 * horton.h's file-level docs), but with map-shaped signatures: `operator()`
 * takes no time argument (maps advance by iteration, not physical time) and
 * separately tracks a wrapped/periodic state alongside an optional
 * unwrapped one (see StandardMap::operator() below); `MapTraits<T>` takes
 * the place of `SystemTraits<T>` and additionally names which state
 * component the generic MSD solver should measure.
 */
#include <cmath>
#include "config.h"

/** @brief Forward declaration; specialized per map type (e.g. below, for StandardMap). */
template <typename MapType>
struct MapTraits;

/** @brief Physical parameters for StandardMap: the stochasticity parameter K. */
struct StandardMapParams {
    double K;
};

/**
 * @brief The Chirikov Standard Map, modified to model transport rather
 * than bounded angle-momentum dynamics: the momentum coordinate is
 * deliberately left unwrapped (see operator() below) so it can accumulate
 * without bound, the same way an unwrapped position accumulates for an ODE
 * system. State layout: `(p, theta)`.
 */
struct StandardMap {
    /**
     * @brief Advances the map by one iteration in place:
     * `p_{n+1} = p_n + K*sin(theta_n)`, `theta_{n+1} = theta_{n+1} mod 2*pi`.
     *
     * Two parallel copies of the state are threaded through every map
     * solver (see e.g. map_msd.cuh): `state_map`, whose angle is always
     * wrapped into `[0, 2*pi)` (needed for `sin(theta)` to stay
     * well-conditioned over many iterations) and whose momentum is left
     * unwrapped (deliberately -- see the class docs above); and
     * `state_unwrapped`, which accumulates the *true* displacement in both
     * components for solvers that need it (e.g. MSD). Escape-only callers
     * that don't need the true displacement pass `nullptr` for
     * `state_unwrapped` and it's simply skipped.
     *
     * @tparam DIMS State dimensionality (2 for StandardMap).
     * @param state_map [in,out] The map's own (angle-wrapped) state, `(p, theta)`.
     * @param state_unwrapped [in,out] True unwrapped `(p, theta)`, or `nullptr` if not needed.
     * @param params Physical parameters.
     */
    template <int DIMS>
    __host__ __device__ void operator()(
        double state_map[DIMS],
        double state_unwrapped[DIMS], // Can be nullptr
        const StandardMapParams& params
    ) const {
        // --- Physics of the Standard Map ---
        double p_update = params.K * sin(state_map[1]);

        // --- Unwrapped State Update ---
        // Only update the unwrapped state if a valid pointer is provided.
        if (state_unwrapped != nullptr) {
            state_unwrapped[0] = state_unwrapped[0] + p_update;
            // The map's periodic momentum is used to unwrap the angle
            double p_map_for_unwrap = fmod(state_map[0] + p_update + M_PI, 2.0 * M_PI) - M_PI;
            state_unwrapped[1] = state_unwrapped[1] + p_map_for_unwrap;
        }

        // --- Map State Update (always happens) ---
        // The map's momentum is intentionally left unwrapped (see the
        // class docs above) -- it's meant to accumulate without bound, the
        // same way check_escape() below relies on it doing.
        state_map[0] = state_map[0] + p_update;
        // The angle is updated with the new wrapped momentum
        state_map[1] = fmod(state_map[1] + state_map[0], 2.0 * M_PI);
        if (state_map[1] < 0) {
            state_map[1] += 2.0 * M_PI;
        }
    }

    /**
     * @brief Jacobian of the map, `J = [[1, K*cos(theta)], [1, 1+K*cos(theta)]]`.
     * Used only by the Lyapunov exponent solver (calculate_lyapunov_exponent()
     * in map_lyapunov.cuh).
     *
     * @tparam DIMS State dimensionality (2 for StandardMap).
     * @param state_map Current (wrapped) state, `(p, theta)`.
     * @param params Physical parameters.
     * @param J_out [out] The `DIMS x DIMS` Jacobian, flattened row-major.
     */
    template <int DIMS>
    __host__ __device__ void jacobian(
        const double state_map[DIMS],
        const StandardMapParams& params,
        double J_out[DIMS * DIMS]
    ) const {
        // J = | 1    K*cos(theta) |
        //     | 1    1+K*cos(theta)|
        double K_cos_theta = params.K * cos(state_map[1]);
        J_out[0] = 1.0;
        J_out[1] = K_cos_theta;
        J_out[2] = 1.0;
        J_out[3] = 1.0 + K_cos_theta;
    }
};

/**
 * @brief StandardMap's trait specialization: escape criterion plus which
 * state component the generic MSD solver measures. Every `MapTraits<T>`
 * specialization must provide exactly these two static members.
 */
template<>
struct MapTraits<StandardMap> {
    /** Tells the generic MSD solver to measure displacement of state[0] (momentum p). */
    static const int msd_dimension_index = 0;

    /**
     * @brief Escape test, called after every operator() call.
     * @param state_map Current (wrapped) state, `(p, theta)`.
     * @return `0.0` if not escaped, else a nonzero basin ID: `1.0` if
     * momentum p has random-walked past `pi` ("escaped up" -- only
     * possible above the chaos threshold K_c ~ 0.9716, since p is left
     * unwrapped, see the class docs above), `-1.0` for `p < -pi`.
     */
    __host__ __device__ static double check_escape(const double state_map[2]) {
        // Escape if momentum p (state_map[0]) goes above a certain threshold
        if (state_map[0] > M_PI) {
            return 1.0; // Escaped through upper boundary
        }
        if (state_map[0] < -M_PI) {
            return -1.0; // Escaped through lower boundary
        }
        return 0.0; // No escape
    }
};

/** @brief Builds StandardMapParams from a resolved config (see config.h). */
template <>
inline StandardMapParams load_params_for<StandardMapParams>(const Config& cfg) {
    StandardMapParams p;
    p.K = cfg.get_double("K", 0.5);
    return p;
}

