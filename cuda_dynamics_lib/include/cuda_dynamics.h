#pragma once // File: cuda_dynamics_lib/include/cuda_dynamics.h

#include <highfive/H5File.hpp>
#include <cuda_runtime.h>
#include <vector>      // Needed for std::vector
#include <numeric>     // Needed for std::accumulate (optional, but good practice)
#include <stdexcept>   // Needed for std::runtime_error

// A utility macro for error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// The GridSetup helper is now part of the library's public interface.
struct GridSetup {
    // --- Member Variables ---
    const int DIMS;
    std::vector<int> grid_res;
    long long num_particles;
    double* h_initial_conditions;

    // --- Constructor: Does all the setup work ---
    GridSetup(int dimensions, const std::vector<int>& resolution,
              const std::vector<double>& min_bounds, const std::vector<double>& max_bounds)
        : DIMS(dimensions) {

        if (resolution.size() != DIMS || min_bounds.size() != DIMS || max_bounds.size() != DIMS) {
            throw std::runtime_error("Dimension mismatch in GridSetup constructor.");
        }
        this->grid_res = resolution;

        this->num_particles = 1;
        for (int res : this->grid_res) { this->num_particles *= res; }

        this->h_initial_conditions = new double[this->num_particles * this->DIMS];

        std::vector<int> current_indices(this->DIMS, 0);
        generate_grid_recursive(0, current_indices, min_bounds, max_bounds);
    }

    // --- Destructor: Cleans up allocated memory ---
    ~GridSetup() {
        delete[] h_initial_conditions;
    }

private:
    void generate_grid_recursive(int dim_idx, std::vector<int>& indices,
                                 const std::vector<double>& min_b, const std::vector<double>& max_b) {
        if (dim_idx == this->DIMS) {
            long long flat_idx = 0;
            long long stride = 1;
            for (int i = this->DIMS - 1; i >= 0; --i) {
                flat_idx += indices[i] * stride;
                stride *= this->grid_res[i];
            }
            for (int j = 0; j < this->DIMS; ++j) {
                h_initial_conditions[flat_idx * this->DIMS + j] =
                    min_b[j] + (max_b[j] - min_b[j]) * indices[j] / (double)(this->grid_res[j] - 1);
            }
        } else {
            for (int i = 0; i < this->grid_res[dim_idx]; ++i) {
                indices[dim_idx] = i;
                generate_grid_recursive(dim_idx + 1, indices, min_b, max_b);
            }
        }
    }
};



inline void save_to_h5(const std::string& filename, const std::string& dset_name, const std::vector<size_t>& dims, const double* data) {
    HighFive::File file(filename, HighFive::File::OpenOrCreate);
    HighFive::DataSet dataset = file.createDataSet<double>(dset_name, HighFive::DataSpace(dims));
    dataset.write_raw(data);
}

inline void save_displacement_components(const std::string& filename, const std::string& dset_name, 
                                  const GridSetup& grid, const double* data) {
    try {
        // Open the file in OpenOrCreate mode to add datasets without overwriting
        HighFive::File file(filename, HighFive::File::OpenOrCreate);

        // 1. Construct the multi-dimensional shape for the dataset.
        // Start with the grid resolution (e.g., {512, 512})
        std::vector<size_t> dims(grid.grid_res.begin(), grid.grid_res.end());
        
        // 2. Append the number of components (DIMS) as the last dimension.
        // The final shape becomes {512, 512, 2} for a 2D system.
        dims.push_back(grid.DIMS);

        // 3. Create the dataset with the correct multi-dimensional shape
        HighFive::DataSet dataset = file.createDataSet<double>(dset_name, HighFive::DataSpace(dims));
        
        // 4. Write the raw, flattened data. HighFive handles the reshaping.
        dataset.write_raw(data);

    } catch (const HighFive::Exception& e) {
        std::cerr << "HDF5 Error while saving component data: " << e.what() << std::endl;
    }
}


// Performs matrix-vector multiplication: v_out = J * v_in
template <int DIMS>
__device__ inline void matrix_vector_mult(const double J[DIMS*DIMS], const double v_in[DIMS], double v_out[DIMS]) {
    for (int i = 0; i < DIMS; ++i) {
        v_out[i] = 0.0;
        for (int j = 0; j < DIMS; ++j) {
            v_out[i] += J[i * DIMS + j] * v_in[j];
        }
    }
}

// Calculates the Euclidean norm (magnitude) of a vector
template <int DIMS>
__device__ inline double vector_norm(const double v[DIMS]) {
    double norm_sq = 0.0;
    for (int i = 0; i < DIMS; ++i) {
        norm_sq += v[i] * v[i];
    }
    return sqrt(norm_sq);
}

// Normalizes a vector in-place
template <int DIMS>
__device__ inline void normalize_vector(double v[DIMS], double norm) {
    if (norm > 1e-12) { // Avoid division by zero
        for (int i = 0; i < DIMS; ++i) {
            v[i] /= norm;
        }
    }
}


template <typename MapType>
struct MapTraits;


// =============================================================================
// == Public Function Declarations for GPU Solvers
// =============================================================================

// The public signature of the host function remains the same. The changes
// are internal to the kernel and the map's definition.
template <int DIMS, typename MapType, typename ParamsType>
void calculate_msd_and_displacement(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    const double* min_bounds,
    const double* max_bounds,
    long long num_particles,
    int num_iterations,
    // Output arrays (host pointers)
    double* h_total_displacement,
    double* h_displacements,
    double* h_msd
);

template <int DIMS, typename MapType, typename ParamsType>
void calculate_lyapunov_exponent(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_iterations,
    // Output array (host pointer)
    double* h_lyapunov_exponents
);

template <int DIMS, typename MapType, typename ParamsType>
void calculate_escape_time(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int max_iterations,
    // Output arrays (host pointers)
    double* h_escape_times,
    double* h_escape_basins
);

// This function is implemented inline here in the header.
template <int DIMS, typename MapType, typename ParamsType>
inline void calculate_phase_space(
    const MapType& map_functor,
    const ParamsType& params,
    const double* h_initial_conditions,
    long long num_particles,
    int num_iterations,
    double* h_phase_space_out // Output array for all trajectories
) {
    // Loop over each particle's initial condition
    for (long long i = 0; i < num_particles; ++i) {
        
        double state_map[DIMS];
        // Initialize the state for the current particle
        for (int j = 0; j < DIMS; ++j) {
            state_map[j] = h_initial_conditions[i * DIMS + j];
        }

        // Loop over the iterations to evolve this single particle
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Store the current state in the large output array before evolving
            for (int j = 0; j < DIMS; ++j) {
                // The memory layout is (particle, iteration, dimension)
                h_phase_space_out[(i * num_iterations + iter) * DIMS + j] = state_map[j];
            }

            // Evolve the state by one step using the map functor.
            map_functor.template operator()<DIMS>(state_map, nullptr, params);
        }
    }
}


