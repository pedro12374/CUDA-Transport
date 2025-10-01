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
    const double TOTAL_STEPS = 10e6;
    

    // --- Define A2 values to simulate ---
    std::vector<double> a2_values = {0.0,0.005, 0.013, 0.026};
    std::vector<double> a3_values = {0.0,0.005, 0.013};

    // --- Use a sparser grid for trajectory plotting ---
    GridSetup grid(DIMS, {512, 512}, {-M_PI, -2*M_PI}, {M_PI, 2*M_PI});
    HortonSystem system;

    namespace fs = std::filesystem;
    fs::create_directory("dat");
    std::string output_file = "dat/horton_escape_A2_A3.h5";
    if (fs::exists(output_file)) {
        fs::remove(output_file);
    }

    // --- Main Calculation Loop ---
    for (double a2 : a2_values) {
        for(double a3: a3_values){
        HortonSystemParams params = {
            .A1 = 0.026, .A2 = a2, .A3 = a3,
            .kx1 = 6.0, .ky1 = 3.0, .w1 = 0.476,
            .kx2 = -3.5, .ky2 = -1.5, .w2 = 0.476,
            .kx3 = -2.5, .ky3 = -1.5, .w3 = 0.476,
        };
        params.v2 = fabs(params.w2/params.ky2 - params.w1/params.ky1);
        params.v3 = fabs(params.w3/params.ky3 - params.w1/params.ky1);
       

         std::cout << "\n--- Running Escape Analysis for A2=" << a2 << " ---" << "A3 = "<< a3  << std::endl;

        // --- Allocate memory for BOTH outputs ---
        double* h_escape_times = new double[grid.num_particles];
        double* h_escape_basins = new double[grid.num_particles];

            // Call the library function
            calculate_ode_escape<DIMS, HortonSystem, HortonSystemParams>(
            system, params, grid.h_initial_conditions, grid.num_particles,
            TOTAL_STEPS, DT, h_escape_times, h_escape_basins);

             std::stringstream base_name;
        base_name << "A2_" << std::fixed << std::setprecision(4) << a2<< "_A3_" << std::fixed << std::setprecision(4) << a3;
        std::string time_dset_name = "EscapeTime_" + base_name.str();
        std::string basin_dset_name = "EscapeBasin_" + base_name.str();
        
        std::vector<size_t> grid_dims = {(size_t)grid.grid_res[0], (size_t)grid.grid_res[1]};
        save_to_h5(output_file, time_dset_name, grid_dims, h_escape_times);
        save_to_h5(output_file, basin_dset_name, grid_dims, h_escape_basins);
        
        std::cout << "Saved Escape Time and Basin data." << std::endl;

        delete[] h_escape_times;
        delete[] h_escape_basins;
    }
    }
    std::cout << "\nAll stroboscopic simulations complete." << std::endl;
    return 0;
}