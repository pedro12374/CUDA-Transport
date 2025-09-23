#pragma once
#include <cmath>

// Forward declaration of the MapTraits struct
template <typename MapType>
struct MapTraits;

// Parameter struct for the Standard Map
struct StandardMapParams {
    double K;
};

// The Standard Map functor
struct StandardMap {
    // This operator evolves the state by one step.
    // It is now marked as __host__ __device__ to be callable from both CPU and GPU.
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
        // The map's internal momentum is always wrapped
        state_map[0] = state_map[0] + p_update;//fmod(state_map[0] + p_update + M_PI, 2.0 * M_PI) - M_PI;
        // The angle is updated with the new wrapped momentum
        state_map[1] = fmod(state_map[1] + state_map[0], 2.0 * M_PI);
        if (state_map[1] < 0) {
            state_map[1] += 2.0 * M_PI;
        }
    }

    // This method provides the Jacobian matrix for Lyapunov calculation.
    // It is now also marked as __host__ __device__.
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

// Trait specialization for the Standard Map
template<>
struct MapTraits<StandardMap> {
    // Tells the generic solver to measure the MSD of the first coordinate (momentum)
    static const int msd_dimension_index = 0;

    // NEW: Function to check for escape conditions
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

