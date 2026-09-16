#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "maps/horton.h"

int main(int argc, char** argv) {
    const int DIMS = 2;
    const double DT = 0.01;
     // Time between snapshots

    // Final integration time, in the same units as DT. Defaults to 1e5
    // (10e4) unless overridden on the command line, e.g.:
    //   ./horton_msd 1e4
    double FINAL_TIME = 10e4;
    if (argc > 1) {
        FINAL_TIME = std::atof(argv[1]);
    }

    // Calculate the total number of steps automatically
    const int TOTAL_STEPS = static_cast<int>(FINAL_TIME / DT);
    std::cout << "FINAL_TIME = " << FINAL_TIME << " (TOTAL_STEPS = " << TOTAL_STEPS << ")" << std::endl;

    // Number of log-spaced MSD samples to keep per (A2,A3) pair, instead of
    // one value per step (which at 1e7 steps would be ~80MB per pair).
    const int NUM_MSD_SAMPLES = 300;

    // --- Define A2 values to simulate ---
    std::vector<double> a2_values = {0.0,0.1, 0.5, 1.0};
    std::vector<double> a3_values = {0.0,0.1, 0.5};

    // --- Use a sparser grid for trajectory plotting ---
    GridSetup grid(DIMS, {1024, 1024}, {-M_PI, -2*M_PI}, {M_PI, 2*M_PI});
    HortonSystem system;

    namespace fs = std::filesystem;
    fs::create_directory("dat");
    std::string output_file = "dat/horton_msd_A2_A3.h5";
    if (fs::exists(output_file)) {
        fs::remove(output_file);
    }

    // The log-spaced sample times are the same for every (A2,A3) pair (same
    // TOTAL_STEPS/DT/NUM_MSD_SAMPLES), so save them once as a shared axis
    // instead of duplicating them per dataset.
    std::vector<int> msd_sample_steps = generate_log_spaced_steps(TOTAL_STEPS, NUM_MSD_SAMPLES);
    std::vector<double> msd_sample_times(msd_sample_steps.size());
    for (size_t k = 0; k < msd_sample_steps.size(); ++k) {
        msd_sample_times[k] = static_cast<double>(msd_sample_steps[k]) * DT;
    }
    std::vector<size_t> msd_time_dims = {msd_sample_times.size()};
    save_to_h5(output_file, "MSD_sample_times", msd_time_dims, msd_sample_times.data());

    // --- Main Calculation Loop ---
    for (double a2 : a2_values) {
        for(double a3: a3_values){
        HortonSystemParams params = {
            .A1 = 1.0, .A2 = a2, .A3 = a3,
            .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
            .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
            .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
        };
        params.v2 = fabs(params.w2/params.ky2 - params.w1/params.ky1);
        params.v3 = fabs(params.w3/params.ky3 - params.w1/params.ky1);

        std::cout << "\n--- Running Analysis for A2=" << a2 << ", A3=" << a3 << " ---" << std::endl;

        std::vector<double> h_total_displacement(grid.num_particles);
            std::vector<double> h_displacements(grid.num_particles * DIMS);
            std::vector<double> h_msd;
            std::vector<double> h_msd_sample_times; // same for every pair; already saved once above

            calculate_ode_msd_and_displacement<DIMS, HortonSystem, HortonSystemParams>(
                system, params, grid.h_initial_conditions.data(), grid.num_particles,
                TOTAL_STEPS, DT, h_total_displacement.data(), h_displacements.data(),
                h_msd, h_msd_sample_times, NUM_MSD_SAMPLES);

            // --- Create unique dataset names ---
            std::stringstream base_dset_name;
            base_dset_name << "A2_" << std::fixed << std::setprecision(4) << a2
                             << "_A3_" << std::fixed << std::setprecision(4) << a3;
            
            std::string msd_dset_name = "MSD_" + base_dset_name.str();
            std::string disp_dset_name = "Displacement_" + base_dset_name.str();
            // ADD a dataset name for total displacement
            std::string total_disp_dset_name = "TotalDisplacement_" + base_dset_name.str();

            // --- Save All Data ---
            std::vector<size_t> grid_dims_2d = {(size_t)grid.grid_res[0], (size_t)grid.grid_res[1]};
            std::vector<size_t> msd_dims_1d = {h_msd.size()};

            save_to_h5(output_file, msd_dset_name, msd_dims_1d, h_msd.data());
            save_displacement_components(output_file, disp_dset_name, grid, h_displacements.data());

            // ADD THIS LINE to save the total displacement
            save_to_h5(output_file, total_disp_dset_name, grid_dims_2d, h_total_displacement.data());

            std::cout << "Saved MSD, Directional, and Total Displacement data." << std::endl;
    }
    }
    std::cout << "\nAll MSD simulations complete." << std::endl;
    return 0;
}