#pragma once
/**
 * @file cuda_dynamics.h
 * @brief Core library header: initial-condition grid setup (GridSetup),
 * HDF5 output (save_to_h5(), save_displacement_components()), the shared
 * RK4 integrator (rk4_step_t()) every ODE solver builds on, and small
 * device-side vector/matrix helpers. Pulls in every solver header
 * (solvers/*.cuh) at the bottom, so including this one header is enough to
 * use the whole library.
 *
 * Most users won't need to call anything here directly -- see runner.cuh
 * for the config-driven entry points (run_ode_generic()/run_map_generic())
 * that most drivers use instead. This header matters directly to: (a)
 * anyone writing a new dynamical system (GridSetup, and the
 * SystemTraits/MapTraits forward declarations your system specializes --
 * see maps/horton.h), and (b) anyone calling a solver's calculate_*()
 * function by hand instead of through the generic runner.
 */

#include <H5Cpp.h>
#include <cuda_runtime.h>
#include <vector>      // Needed for std::vector
#include <numeric>     // Needed for std::accumulate (optional, but good practice)
#include <stdexcept>   // Needed for std::runtime_error
#include <filesystem>  // Needed to check for an existing HDF5 file
#include <iostream>    // Needed for std::cerr

template <typename SystemType>
struct SystemTraits;

// A utility macro for error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief Builds a regular, DIMS-dimensional tensor-product grid of initial
 * conditions -- the standard way this library's drivers populate the
 * `h_initial_conditions` array every solver's `calculate_*()` function
 * takes. Constructed once by `run_ode_generic()`/`run_map_generic()`
 * (runner.cuh) from a config file's `grid_dims`/`grid_min`/`grid_max`.
 *
 * A resolution-1 dimension (`grid_res[j] == 1`) fixes that coordinate at
 * `min_bounds[j]` rather than being an error -- the standard way to hold
 * some state components fixed (e.g. momenta) while scanning others (e.g.
 * position), such as Henon-Heiles' "release from rest" escape-basin scan
 * (see maps/henon_heiles.h and configs/henon_heiles_escape.cfg).
 */
struct GridSetup {
    /** State dimensionality (must match every array this grid is passed to). */
    const int DIMS;
    /** Number of grid points along each dimension, length DIMS. */
    std::vector<int> grid_res;
    /** Total particle count: the product of grid_res. */
    long long num_particles;
    /** Flattened initial conditions, length `num_particles * DIMS`, layout `(particle, dimension)`. */
    std::vector<double> h_initial_conditions;

    /**
     * @param dimensions State dimensionality.
     * @param resolution Grid points per dimension, length `dimensions`.
     * @param min_bounds Lower bound per dimension, length `dimensions`.
     * @param max_bounds Upper bound per dimension, length `dimensions`
     * (ignored for any dimension with `resolution[j] == 1`; only
     * `min_bounds[j]` is used for that dimension's fixed value).
     * @throws std::runtime_error if `resolution`/`min_bounds`/`max_bounds`
     * don't all have length `dimensions`.
     */
    GridSetup(int dimensions, const std::vector<int>& resolution,
              const std::vector<double>& min_bounds, const std::vector<double>& max_bounds)
        : DIMS(dimensions) {

        if (resolution.size() != DIMS || min_bounds.size() != DIMS || max_bounds.size() != DIMS) {
            throw std::runtime_error("Dimension mismatch in GridSetup constructor.");
        }
        this->grid_res = resolution;

        this->num_particles = 1;
        for (int res : this->grid_res) { this->num_particles *= res; }

        this->h_initial_conditions.resize(this->num_particles * this->DIMS);

        std::vector<int> current_indices(this->DIMS, 0);
        generate_grid_recursive(0, current_indices, min_bounds, max_bounds);
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
                // A single-point dimension (grid_res[j] == 1) fixes that
                // coordinate at min_b[j] -- e.g. scanning (x0,y0) at a fixed
                // (px0,py0) for a Hamiltonian system. Guards the division
                // below, which would otherwise be 0/0.
                h_initial_conditions[flat_idx * this->DIMS + j] = (this->grid_res[j] > 1)
                    ? min_b[j] + (max_b[j] - min_b[j]) * indices[j] / (double)(this->grid_res[j] - 1)
                    : min_b[j];
            }
        } else {
            for (int i = 0; i < this->grid_res[dim_idx]; ++i) {
                indices[dim_idx] = i;
                generate_grid_recursive(dim_idx + 1, indices, min_b, max_b);
            }
        }
    }
};



// Opens filename for appending a new dataset if it already exists, or
// creates it fresh otherwise (mirrors HighFive::File::OpenOrCreate).
inline H5::H5File open_or_create_h5(const std::string& filename) {
    if (std::filesystem::exists(filename)) {
        return H5::H5File(filename, H5F_ACC_RDWR);
    }
    return H5::H5File(filename, H5F_ACC_TRUNC);
}

/**
 * @brief Writes `data` to a new dataset in an HDF5 file, creating the file
 * if it doesn't exist, or adding the dataset to it (without disturbing
 * existing datasets) if it does -- so a driver can call this once per
 * output array across a parameter sweep and accumulate everything into one
 * file.
 * @param filename HDF5 file path. Parent directory must already exist.
 * @param dset_name Name for the new dataset. Fails if a dataset with this
 * name already exists in `filename`.
 * @param dims Dataset shape.
 * @param data Row-major data, exactly `dims[0]*dims[1]*...` doubles.
 * @throws std::runtime_error on any HDF5 error (file/dataset creation, write).
 */
inline void save_to_h5(const std::string& filename, const std::string& dset_name, const std::vector<size_t>& dims, const double* data) {
    H5::Exception::dontPrint();
    try {
        H5::H5File file = open_or_create_h5(filename);
        std::vector<hsize_t> h5_dims(dims.begin(), dims.end());
        H5::DataSpace dataspace(static_cast<int>(h5_dims.size()), h5_dims.data());
        H5::DataSet dataset = file.createDataSet(dset_name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(data, H5::PredType::NATIVE_DOUBLE);
    } catch (const H5::Exception& e) {
        // H5::Exception doesn't derive from std::exception, so rethrow as
        // something that prints a useful message if left uncaught.
        throw std::runtime_error("HDF5 error while saving dataset '" + dset_name + "': " + e.getDetailMsg());
    }
}

/**
 * @brief Like save_to_h5(), but for a per-component array shaped like the
 * grid itself (e.g. MSD's per-particle displacement, `(x,y)` per grid
 * point): the dataset gets shape `grid.grid_res + [grid.DIMS]`, e.g.
 * `{512, 512, 2}` for a 2D, 512x512 grid, instead of needing the caller to
 * build that shape by hand.
 * @param filename HDF5 file path. Parent directory must already exist.
 * @param dset_name Name for the new dataset.
 * @param grid The grid `data` was computed on (supplies the shape).
 * @param data Row-major data, `num_particles * grid.DIMS` doubles.
 *
 * @note Unlike save_to_h5(), HDF5 errors here are caught and logged to
 * stderr rather than thrown -- a pre-existing inconsistency between the
 * two functions, not deliberate API design; callers shouldn't rely on it.
 */
inline void save_displacement_components(const std::string& filename, const std::string& dset_name,
                                  const GridSetup& grid, const double* data) {
    try {
        H5::Exception::dontPrint();
        // Open (or create) the file so datasets accumulate rather than overwrite.
        H5::H5File file = open_or_create_h5(filename);

        // 1. Construct the multi-dimensional shape for the dataset.
        // Start with the grid resolution (e.g., {512, 512})
        std::vector<hsize_t> dims(grid.grid_res.begin(), grid.grid_res.end());

        // 2. Append the number of components (DIMS) as the last dimension.
        // The final shape becomes {512, 512, 2} for a 2D system.
        dims.push_back(static_cast<hsize_t>(grid.DIMS));

        // 3. Create the dataset with the correct multi-dimensional shape
        H5::DataSpace dataspace(static_cast<int>(dims.size()), dims.data());
        H5::DataSet dataset = file.createDataSet(dset_name, H5::PredType::NATIVE_DOUBLE, dataspace);

        // 4. Write the raw, flattened data.
        dataset.write(data, H5::PredType::NATIVE_DOUBLE);

    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 Error while saving component data: " << e.getDetailMsg() << std::endl;
    }
}


/**
 * @brief Matrix-vector product `v_out = J * v_in`, `J` row-major
 * `DIMS x DIMS`. Used by the Lyapunov exponent solvers to evolve a tangent
 * vector through a system's jacobian().
 */
template <int DIMS>
__device__ inline void matrix_vector_mult(const double J[DIMS*DIMS], const double v_in[DIMS], double v_out[DIMS]) {
    for (int i = 0; i < DIMS; ++i) {
        v_out[i] = 0.0;
        for (int j = 0; j < DIMS; ++j) {
            v_out[i] += J[i * DIMS + j] * v_in[j];
        }
    }
}

/** @brief Euclidean norm of a DIMS-vector. */
template <int DIMS>
__device__ inline double vector_norm(const double v[DIMS]) {
    double norm_sq = 0.0;
    for (int i = 0; i < DIMS; ++i) {
        norm_sq += v[i] * v[i];
    }
    return sqrt(norm_sq);
}

/** @brief Normalizes `v` in place to unit length; a no-op if `norm <= 1e-12` (avoids dividing by ~0). */
template <int DIMS>
__device__ inline void normalize_vector(double v[DIMS], double norm) {
    if (norm > 1e-12) { // Avoid division by zero
        for (int i = 0; i < DIMS; ++i) {
            v[i] /= norm;
        }
    }
}

/**
 * @brief One classical (non-adaptive, 4th-order) Runge-Kutta step,
 * advancing `state` in place from time `t` to `t + dt`. The single
 * integrator every ODE solver in this library (ode_escape.cuh,
 * ode_msd.cuh, ode_lyapunov.cuh, ode_strobo.cuh) is built on -- a new
 * solver for continuous-time systems should use this rather than
 * hand-rolling its own stepper.
 *
 * @tparam DIMS State dimensionality.
 * @tparam SystemType A system type providing `operator()<DIMS>(state,
 * dstate_dt, params, t)` (see maps/horton.h's file-level docs for the full
 * interface a system implements).
 * @tparam ParamsType That system's parameter struct type.
 * @param state [in,out] State to advance in place.
 * @param t Time at the start of the step.
 * @param dt Step size.
 * @param system The system functor (stateless; typically default-constructed by the caller).
 * @param params Physical parameters passed through to every `operator()` call.
 */
template <int DIMS, typename SystemType, typename ParamsType>
__device__ inline void rk4_step_t(
    double state[DIMS],
    double t,
    double dt,
    const SystemType& system,
    const ParamsType& params)
{
    double k1[DIMS], k2[DIMS], k3[DIMS], k4[DIMS];
    double temp_state[DIMS];

    // Calculate k1 at t
    system.template operator()<DIMS>(state, k1, params, t);

    // Calculate k2 at t + dt/2
    for (int i = 0; i < DIMS; ++i) temp_state[i] = state[i] + 0.5 * dt * k1[i];
    system.template operator()<DIMS>(temp_state, k2, params, t + 0.5 * dt);

    // Calculate k3 at t + dt/2
    for (int i = 0; i < DIMS; ++i) temp_state[i] = state[i] + 0.5 * dt * k2[i];
    system.template operator()<DIMS>(temp_state, k3, params, t + 0.5 * dt);

    // Calculate k4 at t + dt
    for (int i = 0; i < DIMS; ++i) temp_state[i] = state[i] + dt * k3[i];
    system.template operator()<DIMS>(temp_state, k4, params, t + dt);

    // Update state
    for (int i = 0; i < DIMS; ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}


template <typename MapType>
struct MapTraits;

/**
 * @brief Splits `[0, total)` into consecutive batches of at most
 * `batch_size` steps, calling `launch_batch(step_offset, steps_this_batch)`
 * once per batch. Used by the batched escape/MSD solvers (ode_escape.cuh,
 * ode_msd.cuh) to avoid running an entire long integration as one kernel
 * launch -- see either file's top comment for why that matters on a
 * shared GPU. Factored out here so both solvers share one implementation
 * (and one validation) instead of two independently-maintained copies.
 *
 * @param total Total number of steps to cover.
 * @param batch_size Steps per batch.
 * @param launch_batch Called as `launch_batch(step_offset, steps_this_batch)`
 * for each batch, in order, covering `[0, total)` exactly once.
 * @throws std::invalid_argument if `batch_size <= 0` -- silently looping
 * forever (steps_done never advancing, or regressing) is worse than a
 * clear error naming the actual mistake.
 */
template <typename LaunchBatchFn>
inline void run_in_batches(int total, int batch_size, LaunchBatchFn&& launch_batch) {
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "run_in_batches: batch_size must be > 0, got " + std::to_string(batch_size));
    }
    int steps_done = 0;
    while (steps_done < total) {
        int this_batch = (total - steps_done < batch_size) ? (total - steps_done) : batch_size;
        launch_batch(steps_done, this_batch);
        steps_done += this_batch;
    }
}


// =============================================================================
// == Public Function Declarations for GPU Solvers
// =============================================================================



/**
 * @brief Records every iterate of `map_functor` for `num_iterations`, for
 * every particle -- a full phase-space trajectory, not just an escape
 * time/basin. Driven by `calculation = phasespace` in a map system's config
 * (see run_map_generic() in runner.cuh).
 *
 * @note **CPU-only.** Unlike every other `calculate_*()` function in this
 * library, there's no GPU kernel for this one -- it's a plain host loop.
 * Fine for the grid sizes/iteration counts a phase-space plot typically
 * needs; slow for anything approaching the particle counts used for escape
 * scans.
 *
 * @tparam DIMS State dimensionality.
 * @tparam MapType A map type providing `operator()<DIMS>(state_map,
 * state_unwrapped, params)` (see maps/standard_map.h for the full map
 * interface).
 * @tparam ParamsType That map's parameter struct type.
 * @param map_functor The map functor (stateless; typically default-constructed by the caller).
 * @param params Physical parameters.
 * @param h_initial_conditions Host array, `num_particles * DIMS` doubles.
 * @param num_particles Particle count.
 * @param num_iterations Iterations to record per particle.
 * @param h_phase_space_out [out] Host array, `num_particles * num_iterations
 * * DIMS` doubles, layout `(particle, iteration, dimension)`.
 */
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


#include "solvers/ode_escape.cuh"
#include "solvers/ode_lyapunov.cuh"
#include "solvers/ode_msd.cuh"
#include "solvers/ode_strobo.cuh"

#include "solvers/map_escape.cuh"
#include "solvers/map_lyapunov.cuh"
#include "solvers/map_msd.cuh"