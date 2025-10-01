#pragma once
#include <cmath>

template <typename SystemType>
struct SystemTraits;
// Parameter struct for the Three-Wave System
struct HortonSystemParams {
    // Amplitudes
    double A1, A2, A3;

    // Wave vectors and frequencies
    double kx1, ky1, w1;
    double kx2, ky2, w2;
    double kx3, ky3, w3;

    // Phase velocities (calculated from the above)
    double v2, v3;
};

// The Three-Wave System functor
struct HortonSystem {
    // This operator calculates the derivatives (dx/dt, dy/dt)
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
        double term2_x = params.A2 * params.ky2 * sin(params.kx2 * x + 1.0) * sin(arg2_y);
        double term2_y = params.A2 * params.kx2 * cos(params.kx2 * x + 1.0) * cos(arg2_y);

        // Term 3
        double arg3_y = params.ky3 * (y - params.v3 * t);
        double term3_x = params.A3 * params.ky3 * sin(params.kx3 * x + 1.0) * sin(arg3_y);
        double term3_y = params.A3 * params.kx3 * cos(params.kx3 * x + 1.0) * cos(arg3_y);

        dstate_dt[0] = term1_x + term2_x + term3_x; // dxdt
        dstate_dt[1] = term1_y + term2_y + term3_y; // dydt

        

    }

    // This method provides the Jacobian matrix for Lyapunov calculation.
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
        
        double cos_kx2_x_phase = cos(params.kx2 * x + M_PI);
        double sin_kx2_x_phase = sin(params.kx2 * x + M_PI);
        double arg2_y = params.ky2 * (y - params.v2 * t);
        double cos_arg2_y = cos(arg2_y);
        double sin_arg2_y = sin(arg2_y);

        double cos_kx3_x_phase = cos(params.kx3 * x + M_PI);
        double sin_kx3_x_phase = sin(params.kx3 * x + M_PI);
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

template<>
struct SystemTraits<HortonSystem> {

        __host__ __device__ static void post_step_update(double state[2]) {
        // Wrap the 'y' coordinate to be periodic in [-PI, PI]
        // This is the crucial physics you identified!
        state[1] = fmod(state[1] + 2.0 * M_PI, 4.0 * M_PI) - 2.0 * M_PI;
    }

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