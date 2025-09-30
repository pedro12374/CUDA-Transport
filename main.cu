#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "maps/horton.h"

int main() {
    const int DIMS = 2;
    const double DT = 0.01;
     // Time between snapshots
    const int STROBOSCOPIC_POINTS = 500;   // Number of points per orbit

    // --- Define A2 values to simulate ---
    std::vector<double> a2_values = {0.1, 0.5, 1.0, 1.5};
    std::vector<double> a3_values = {0.1, 0.5, 1.0, 1.5};

    // --- Use a sparser grid for trajectory plotting ---
    GridSetup grid(DIMS, {30, 30}, {-M_PI, -2*M_PI}, {M_PI, 2*M_PI});
    ThreeWaveSystem system;

    namespace fs = std::filesystem;
    fs::create_directory("dat");
    std::string output_file = "dat/three_wave_stroboscopic_A2_A3.h5";
    if (fs::exists(output_file)) {
        fs::remove(output_file);
    }

    // --- Main Calculation Loop ---
    for (double a2 : a2_values) {
        for(double a3: a3_values){
        ThreeWaveSystemParams params = {
            .A1 = 1.0, .A2 = a2, .A3 = a3,
            .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
            .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
            .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
        };
        params.v2 = fabs(params.w2/params.ky2 - params.w1/params.ky1);
        params.v3 = fabs(params.w3/params.ky3 - params.w1/params.ky1);
        const double STROBOSCOPIC_TAU = 2.0 * M_PI/params.v2;

        std::cout << "\n--- Running Analysis for A2=" << a2 << ", A3=" << a3 << " ---" << std::endl;

        double* h_strobo_map = new double[grid.num_particles * STROBOSCOPIC_POINTS * DIMS];

        calculate_ode_stroboscopic_map<DIMS, ThreeWaveSystem, ThreeWaveSystemParams>(
            system, params, grid.h_initial_conditions, grid.num_particles,
            STROBOSCOPIC_POINTS, STROBOSCOPIC_TAU, DT, h_strobo_map);

        std::stringstream dset_name_stream;
        dset_name_stream << "Strobo_A2_" << std::fixed << std::setprecision(2) << a2
                             << "_A3_" << std::fixed << std::setprecision(2) << a3;
            std::string dset_name = dset_name_stream.str();
        
        std::vector<size_t> dims = {(size_t)grid.num_particles, (size_t)STROBOSCOPIC_POINTS, (size_t)DIMS};
        save_to_h5(output_file, dset_name, dims, h_strobo_map);
        std::cout << "Saved data to dataset: " << dset_name << std::endl;

        delete[] h_strobo_map;
    }
    }
    std::cout << "\nAll stroboscopic simulations complete." << std::endl;
    return 0;
}