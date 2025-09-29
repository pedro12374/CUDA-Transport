#include <iostream>
#include <string>
#include <math.h>
#include <filesystem>
// Just include your library's header. It now contains GridSetup.
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "maps/standard_map.h" // Still need the specific map definition



int main() {
 
     const int DIMS = 2;
    const int NUM_ITERATIONS = 10e6; // Number of points per orbit
    // Define a vector of K values to analyze
    std::vector<double> K_values = {0.5, 0.971635, 1.5, 6.47};

    std::cout << "--- CPU Phase Space Calculation ---" << std::endl;

    // --- System Configuration (done once) ---
    GridSetup grid(DIMS, {1024, 1024}, {-M_PI, 0.0}, {M_PI, 2.0 * M_PI});
    std::cout << "Configured for " << grid.num_particles 
              << " particles over " << NUM_ITERATIONS << " iterations." << std::endl;

    // --- Directory and File Cleanup (done once) ---
    namespace fs = std::filesystem;
    std::string output_dir = "dat";
    fs::create_directory(output_dir);
    std::string output_file = output_dir + "/Escape.h5";
    if (fs::exists(output_file)) {
        fs::remove(output_file);
    }

        double* h_escape_basins = new double[grid.num_particles];
        double* h_escape_times = new double[grid.num_particles];

    
    StandardMap map;

    // --- Main Calculation Loop for each K value ---
    for (double K : K_values) {
        std::cout << "\nCalculating phase space for K = " << K << "..." << std::endl;
        StandardMapParams params = {K};

        calculate_escape_time<DIMS,StandardMap,StandardMapParams>(
            map, params, grid.h_initial_conditions,
            grid.num_particles, NUM_ITERATIONS,
            h_escape_times, h_escape_basins
        );
        std::cout << "Calculation complete." << std::endl;

        // --- Save Results to HDF5 ---
        std::string dset_name_time = "Time_K_" + std::to_string(K);
        std::string dset_name_basin = "Basin_K_" + std::to_string(K);
        const std::vector<size_t> dims =  std::vector<size_t>(grid.grid_res.begin(), grid.grid_res.end());
        save_to_h5(output_file, dset_name_time, dims, h_escape_times);
        save_to_h5(output_file, dset_name_basin, dims, h_escape_basins);
        }

    // --- Cleanup ---
    delete[] h_escape_times;
    delete[] h_escape_basins;

    return 0;
}