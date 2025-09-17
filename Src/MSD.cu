#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// MODIFIED: Include HighFive header instead of H5Cpp.h
#include <highfive/H5File.hpp>

// =============================================================================
// == CUDA ERROR CHECKING MACRO
// =============================================================================
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// =============================================================================
// == SIMULATION PARAMETERS (Unchanged)
// =============================================================================
#define GRID_DIM_X 1024
#define GRID_DIM_Y 1024
#define BLOCK_SIZE 16
const int NUM_ITERATIONS = 10e3;
const int NUM_PARTICLES = GRID_DIM_X * GRID_DIM_Y;
const double P_MIN = -M_PI;
const double P_MAX =  M_PI;
const double THETA_MIN = 0.0;
const double THETA_MAX = 2.0 * M_PI;

// =============================================================================
// == DATA STRUCTURES (Unchanged)
// =============================================================================
struct Params {
    double K;
};

// =============================================================================
// == HDF5 HELPER FUNCTIONS (MODIFIED FOR HIGHFIVE)
// =============================================================================
// The HighFive API is simpler, but the helper functions are kept for structure.
void write_hdf5_vector(HighFive::File& file, const std::string& dataset_name, const std::vector<size_t>& dims, const double* data) {
    try {
        HighFive::DataSet dataset = file.createDataSet<double>(dataset_name, HighFive::DataSpace(dims));
        dataset.write_raw(data);
    } catch (HighFive::Exception& e) {
        std::cerr << "HighFive Error for dataset '" << dataset_name << "': " << e.what() << std::endl;
    }
}

void write_hdf5_matrix(HighFive::File& file, const std::string& dataset_name, const std::vector<size_t>& dims, const double* data) {
    try {
        HighFive::DataSet dataset = file.createDataSet<double>(dataset_name, HighFive::DataSpace(dims));
        dataset.write_raw(data);
    } catch (HighFive::Exception& e) {
        std::cerr << "HighFive Error for dataset '" << dataset_name << "': " << e.what() << std::endl;
    }
}

// =============================================================================
// == CUDA KERNEL (Unchanged)
// =============================================================================
__global__ void standard_map_msd_kernel(
    double* total_displacement,
    double* displacement_p,
    double* displacement_theta,
    double* msd_over_time,
    Params params,
    int num_iterations)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use the clearer indexing scheme
    int flat_idx = iy * GRID_DIM_X + ix;

    if (ix < GRID_DIM_X && iy < GRID_DIM_Y) {
        const double p0 = P_MIN + (P_MAX - P_MIN) * ix / (double)(GRID_DIM_X - 1);
        const double theta0 = THETA_MIN + (THETA_MAX - THETA_MIN) * iy / (double)(GRID_DIM_Y - 1);

        // --- State Variables ---
        // '_map' variables are for the periodic dynamics of the map itself
        double p_map = p0;
        double theta_map = theta0;

        // '_unwrapped' variables are for measuring true displacement (for MSD and final output)
        double p_unwrapped = p0;
        double theta_unwrapped = theta0;

        for (int i = 0; i < num_iterations; ++i) {
            // 1. Calculate the change in momentum based on the current periodic angle
            double p_update = params.K * sin(theta_map);

            // 2. Update the UNWRAPPED momentum for measurement
            p_unwrapped = p_unwrapped + p_update;
            
            // 3. Update the MAP's momentum and wrap it to the interval [-π, π]
            p_map = fmod(p_map + p_update + M_PI, 2.0 * M_PI) - M_PI;

            // 4. Update the UNWRAPPED angle using the new WRAPPED momentum (as per the toroidal map definition)
            theta_unwrapped = theta_unwrapped + p_map;
            
            // 5. Update the MAP's angle and wrap it to the interval [0, 2π)
            theta_map = fmod(theta_map + p_map, 2.0 * M_PI);
            if (theta_map < 0) {
                theta_map += 2.0 * M_PI;
            }

            // 6. Calculate MSD using ONLY the unwrapped coordinates
            double sq_disp_p = (p_unwrapped - p0) * (p_unwrapped - p0);
            atomicAdd(&msd_over_time[i], sq_disp_p);
        }

        // 7. Calculate final displacement using the final unwrapped values
        double disp_p = p_unwrapped - p0;
        double disp_theta = theta_unwrapped - theta0;
        displacement_p[flat_idx] = disp_p;
        displacement_theta[flat_idx] = disp_theta;
        total_displacement[flat_idx] = sqrt(disp_p * disp_p + disp_theta * disp_theta);
    }
}

// =============================================================================
// == HOST SOLVER FUNCTION (Unchanged)
// =============================================================================
void solver(double K_param, double* h_msd, double* h_total_displacement, double* h_displacement_p, double* h_displacement_theta) {
    // Device Memory Allocation
    double *d_total_displacement, *d_displacement_p, *d_displacement_theta, *d_msd;
    Params* d_params;
    CUDA_CHECK(cudaMalloc(&d_total_displacement, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_displacement_p, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_displacement_theta, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_msd, NUM_ITERATIONS * sizeof(double)));
    // Note: We don't actually need to allocate d_params anymore with this fix, but leaving it doesn't cause harm.
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(Params)));
    CUDA_CHECK(cudaMemset(d_msd, 0, NUM_ITERATIONS * sizeof(double)));

    // Set Up Parameters on the HOST
    Params h_params = {K_param};
    // Note: This cudaMemcpy is also not strictly needed anymore, but is harmless.
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(Params), cudaMemcpyHostToDevice));

    // Kernel Launch
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_DIM_X / BLOCK_SIZE, GRID_DIM_Y / BLOCK_SIZE);
    std::cout << "Calculating for K = " << K_param << "..." << std::endl;
    
    // THE ONLY CHANGE IS HERE: Pass the host struct `h_params` by value
    standard_map_msd_kernel<<<gridSize, blockSize>>>(d_total_displacement, d_displacement_p, d_displacement_theta, d_msd, h_params, NUM_ITERATIONS);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    // Copy Results Back to Host
    CUDA_CHECK(cudaMemcpy(h_total_displacement, d_total_displacement, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_displacement_p, d_displacement_p, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_displacement_theta, d_displacement_theta, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_msd, d_msd, NUM_ITERATIONS * sizeof(double), cudaMemcpyDeviceToHost));

    // Post-processing
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        h_msd[i] /= NUM_PARTICLES;
    }
    std::cout << "MSD normalized." << std::endl;

    // Free Device Memory
    CUDA_CHECK(cudaFree(d_total_displacement));
    CUDA_CHECK(cudaFree(d_displacement_p));
    CUDA_CHECK(cudaFree(d_displacement_theta));
    CUDA_CHECK(cudaFree(d_msd));
    CUDA_CHECK(cudaFree(d_params));
}

// =============================================================================
// == MAIN FUNCTION (MODIFIED FOR HIGHFIVE)
// =============================================================================
int main() {
    std::vector<double> K_values = {0.5, 0.971635, 1.5, 2.5,6.28,6.47};

    // MODIFIED: Using HighFive::File object
    std::cout << "Creating HDF5 output files..." << std::endl;
    HighFive::File msd_file("../dat/msd_p.h5", HighFive::File::Truncate);
    HighFive::File total_disp_file("../dat/total_displacement.h5", HighFive::File::Truncate);
    HighFive::File p_disp_file("../dat/displacement_p.h5", HighFive::File::Truncate);
    HighFive::File theta_disp_file("../dat/displacement_theta.h5", HighFive::File::Truncate);

    // Allocate host memory
    double* h_msd = new double[NUM_ITERATIONS];
    double* h_total_displacement = new double[NUM_PARTICLES];
    double* h_displacement_p = new double[NUM_PARTICLES];
    double* h_displacement_theta = new double[NUM_PARTICLES];

    // Define the dimensions for the datasets
    std::vector<size_t> msd_dims = {(size_t)NUM_ITERATIONS};
    std::vector<size_t> disp_dims = {(size_t)GRID_DIM_Y, (size_t)GRID_DIM_X};

    // Loop through K values, compute, and save
    for (double K : K_values) {
        solver(K, h_msd, h_total_displacement, h_displacement_p, h_displacement_theta);

        std::string dset_name = "K_" + std::to_string(K);
        std::cout << "Saving datasets for " << dset_name << "..." << std::endl;

        write_hdf5_vector(msd_file, dset_name, msd_dims, h_msd);
        write_hdf5_matrix(total_disp_file, dset_name, disp_dims, h_total_displacement);
        write_hdf5_matrix(p_disp_file, dset_name, disp_dims, h_displacement_p);
        write_hdf5_matrix(theta_disp_file, dset_name, disp_dims, h_displacement_theta);
        std::cout << "------------------------------------" << std::endl;
    }

    // Clean up host memory
    delete[] h_msd;
    delete[] h_total_displacement;
    delete[] h_displacement_p;
    delete[] h_displacement_theta;

    std::cout << "\nFinished all computations. Output files are ready." << std::endl;
    return 0;
}