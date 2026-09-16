#pragma once
/**
 * @file runner.cuh
 * @brief Generic, config-file-driven entry points -- run_ode_generic() and
 * run_map_generic() -- that most drivers call from `main()` instead of
 * calling a solver's `calculate_*()` function directly. A new dynamical
 * system needs:
 *   1. A system header (e.g. maps/horton.h) defining the functor + Params
 *      struct + `SystemTraits` (ODE) or `MapTraits` (map) + a
 *      `load_params_for<>` specialization (see config.h) -- the full
 *      "system definition" (see horton.h's file-level docs for the
 *      complete contract, and TUTORIAL.md for a from-scratch walkthrough).
 *   2. A trivial per-system entry point `.cu`, e.g.:
 *      @code
 *        #include "cuda_dynamics_lib/include/cuda_dynamics.h"
 *        #include "cuda_dynamics_lib/include/runner.cuh"
 *        #include "maps/horton.h"
 *        int main(int argc, char** argv) {
 *            return run_ode_generic<2, HortonSystem, HortonSystemParams>(argc, argv);
 *        }
 *      @endcode
 *   3. A config file (see configs/*.cfg for examples) picking what to
 *      calculate, the grid, `dt`/`final_time` (ODE) or `iterations` (map),
 *      and the system's own physical parameters -- optionally as comma-
 *      separated sweep lists (see config.h's enumerate_sweeps()).
 *
 * No solver or library code needs to change to add a system or run a
 * different calculation on it.
 */

#include "cuda_dynamics.h"
#include "config.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace cudat_runner_detail {

inline void prepare_output_file(const std::string& output_file) {
    std::filesystem::path p(output_file);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    if (std::filesystem::exists(output_file)) std::filesystem::remove(output_file);
}

inline std::vector<int> require_dims_list(const Config& cfg, const std::string& key, int DIMS) {
    std::vector<int> v = cfg.get_int_list(key);
    if (static_cast<int>(v.size()) != DIMS) {
        throw std::runtime_error("Config: '" + key + "' must have exactly " +
                                  std::to_string(DIMS) + " entries (one per dimension), got " +
                                  std::to_string(v.size()));
    }
    return v;
}
inline std::vector<double> require_dims_list_d(const Config& cfg, const std::string& key, int DIMS) {
    std::vector<double> v = cfg.get_double_list(key);
    if (static_cast<int>(v.size()) != DIMS) {
        throw std::runtime_error("Config: '" + key + "' must have exactly " +
                                  std::to_string(DIMS) + " entries (one per dimension), got " +
                                  std::to_string(v.size()));
    }
    return v;
}

} // namespace cudat_runner_detail

// =============================================================================
// == Generic runner for continuous-time (RK4-integrated) systems
// =============================================================================

/**
 * @brief Config-driven entry point for a continuous-time (RK4-integrated)
 * system -- reads a config file, builds the initial-condition grid, runs
 * every combination of a parameter sweep, and writes HDF5 output. Intended
 * to be called directly from `main()`; see this header's file-level docs
 * for the ~4-line pattern every `run_<system>.cu` follows.
 *
 * Required config keys (all sweepable except `grid_dims`/`grid_min`/`grid_max`,
 * `calculation`, and `output_file` -- see config.h's enumerate_sweeps()):
 *  - `calculation`: one of `escape`, `msd`, `lyapunov`, `stroboscopic`.
 *  - `output_file`: HDF5 output path; parent directories are created, and
 *    an existing file at this path is deleted before the first write.
 *  - `dt`: integration step size.
 *  - `grid_dims`, `grid_min`, `grid_max`: DIMS entries each (see GridSetup).
 *  - `final_time` (escape/msd/lyapunov): total integration time; steps run
 *    is `(int)(final_time/dt)`.
 *  - `steps_per_batch` (escape/msd, optional, default 20000): see
 *    calculate_ode_escape()/calculate_ode_msd_and_displacement() in
 *    ode_escape.cuh/ode_msd.cuh for why long runs are batched.
 *  - `msd_samples` (msd, optional, default 300): see
 *    generate_log_spaced_steps() in ode_msd.cuh.
 *  - `stroboscopic_tau` (stroboscopic, required), `stroboscopic_points`
 *    (stroboscopic, optional, default 500).
 *  - Any key the system's own `load_params_for<ParamsType>` reads (see
 *    that system's header, e.g. maps/horton.h).
 *
 * HDF5 datasets written, one call per sweep combination, each named with a
 * `_key1_value1_key2_value2...` suffix if anything was swept (empty if
 * not): `EscapeTime`/`EscapeBasin` (escape); `MSD`/`Displacement`/
 * `TotalDisplacement` per combination plus a single shared
 * `MSD_sample_times` (msd); `Lyapunov` (lyapunov); `Strobo` (stroboscopic).
 *
 * @tparam DIMS State dimensionality (fixed at compile time -- this is why
 * each system needs its own tiny entry-point `.cu` rather than being
 * runtime-selectable).
 * @tparam SystemType The system functor type, e.g. `HortonSystem`.
 * @tparam ParamsType That system's parameter struct type.
 * @param argc,argv Command-line arguments as passed to `main()`; expects
 * exactly one argument, the config file path.
 * @return `0` on success; `1` on any error (bad usage, a config problem,
 * or an unknown `calculation` value), after printing a message to stderr.
 */
template <int DIMS, typename SystemType, typename ParamsType>
inline int run_ode_generic(int argc, char** argv) {
    using namespace cudat_runner_detail;
    if (argc < 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "run") << " <config_file>" << std::endl;
        return 1;
    }

    try {
        Config base_cfg = Config::load(argv[1]);

        const std::string calculation = base_cfg.get_string("calculation");
        const std::string output_file = base_cfg.get_string("output_file");
        const double dt = base_cfg.get_double("dt");

        std::vector<int> grid_res = require_dims_list(base_cfg, "grid_dims", DIMS);
        std::vector<double> grid_min = require_dims_list_d(base_cfg, "grid_min", DIMS);
        std::vector<double> grid_max = require_dims_list_d(base_cfg, "grid_max", DIMS);
        GridSetup grid(DIMS, grid_res, grid_min, grid_max);

        SystemType system;
        prepare_output_file(output_file);

        std::vector<SweepResult> sweeps = enumerate_sweeps(base_cfg);
        std::cout << "run_ode_generic: " << sweeps.size() << " run(s), calculation='"
                  << calculation << "', " << grid.num_particles << " particles" << std::endl;

        bool saved_msd_times = false;
        double shared_msd_final_time = 0.0;
        int shared_msd_num_samples = 0;

        for (size_t i = 0; i < sweeps.size(); ++i) {
            const Config& cfg = sweeps[i].config;
            const std::string& suffix = sweeps[i].suffix;
            const std::string dset_suffix = suffix.empty() ? "" : ("_" + suffix);
            ParamsType params = load_params_for<ParamsType>(cfg);

            if (!suffix.empty()) {
                std::cout << "\n--- [" << (i + 1) << "/" << sweeps.size() << "] " << suffix << " ---" << std::endl;
            }

            if (calculation == "escape") {
                const double final_time = cfg.get_double("final_time");
                const int total_steps = static_cast<int>(final_time / dt);
                const int steps_per_batch = cfg.get_int("steps_per_batch", 20000);

                std::vector<double> h_escape_times(grid.num_particles);
                std::vector<double> h_escape_basins(grid.num_particles);
                calculate_ode_escape<DIMS, SystemType, ParamsType>(
                    system, params, grid.h_initial_conditions.data(), grid.num_particles,
                    total_steps, dt, h_escape_times.data(), h_escape_basins.data(), steps_per_batch);

                std::vector<size_t> dims(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "EscapeTime" + dset_suffix, dims, h_escape_times.data());
                save_to_h5(output_file, "EscapeBasin" + dset_suffix, dims, h_escape_basins.data());

            } else if (calculation == "msd") {
                const double final_time = cfg.get_double("final_time");
                const int total_steps = static_cast<int>(final_time / dt);
                const int num_samples = cfg.get_int("msd_samples", 300);
                const int steps_per_batch = cfg.get_int("steps_per_batch", 20000);

                std::vector<double> h_total_disp(grid.num_particles);
                std::vector<double> h_disp(grid.num_particles * DIMS);
                std::vector<double> h_msd, h_msd_t;
                calculate_ode_msd_and_displacement<DIMS, SystemType, ParamsType>(
                    system, params, grid.h_initial_conditions.data(), grid.num_particles,
                    total_steps, dt, h_total_disp.data(), h_disp.data(), h_msd, h_msd_t,
                    num_samples, steps_per_batch);

                // MSD_sample_times is shared across combinations ONLY when
                // final_time/msd_samples are actually the same for all of
                // them (the common case -- sweeping a physical parameter
                // like A2/A3, not the integration/sampling settings). If a
                // later combination's final_time or msd_samples differs
                // (both are legal, if unusual, sweep keys), its sample
                // times are genuinely different and get their own
                // per-combination dataset instead of silently reusing the
                // first combination's, which would otherwise mismatch.
                std::vector<size_t> t_dims = { h_msd_t.size() };
                if (!saved_msd_times) {
                    save_to_h5(output_file, "MSD_sample_times", t_dims, h_msd_t.data());
                    saved_msd_times = true;
                    shared_msd_final_time = final_time;
                    shared_msd_num_samples = num_samples;
                } else if (final_time != shared_msd_final_time || num_samples != shared_msd_num_samples) {
                    save_to_h5(output_file, "MSD_sample_times" + dset_suffix, t_dims, h_msd_t.data());
                }

                std::vector<size_t> msd_dims = { h_msd.size() };
                std::vector<size_t> grid_dims_2d(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "MSD" + dset_suffix, msd_dims, h_msd.data());
                save_displacement_components(output_file, "Displacement" + dset_suffix, grid, h_disp.data());
                save_to_h5(output_file, "TotalDisplacement" + dset_suffix, grid_dims_2d, h_total_disp.data());

            } else if (calculation == "lyapunov") {
                const double final_time = cfg.get_double("final_time");
                const int total_steps = static_cast<int>(final_time / dt);

                std::vector<double> h_lyap(grid.num_particles);
                calculate_ode_lyapunov_exponent<DIMS, SystemType, ParamsType>(
                    system, params, grid.h_initial_conditions.data(), grid.num_particles,
                    total_steps, dt, h_lyap.data());

                std::vector<size_t> dims(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "Lyapunov" + dset_suffix, dims, h_lyap.data());

            } else if (calculation == "stroboscopic") {
                const double tau = cfg.get_double("stroboscopic_tau");
                const int num_points = cfg.get_int("stroboscopic_points", 500);

                std::vector<double> h_strobo(static_cast<size_t>(grid.num_particles) * num_points * DIMS);
                calculate_ode_stroboscopic_map<DIMS, SystemType, ParamsType>(
                    system, params, grid.h_initial_conditions.data(), grid.num_particles,
                    num_points, tau, dt, h_strobo.data());

                std::vector<size_t> dims = { static_cast<size_t>(grid.num_particles),
                                              static_cast<size_t>(num_points),
                                              static_cast<size_t>(DIMS) };
                save_to_h5(output_file, "Strobo" + dset_suffix, dims, h_strobo.data());

            } else {
                std::cerr << "Unknown calculation '" << calculation
                          << "' (expected escape|msd|lyapunov|stroboscopic)" << std::endl;
                return 1;
            }

            std::cout << "Saved '" << calculation << "' results"
                      << (suffix.empty() ? "" : (" for " + suffix)) << "." << std::endl;
        }

        std::cout << "\nAll '" << calculation << "' runs complete." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// =============================================================================
// == Generic runner for discrete maps
// =============================================================================

/**
 * @brief Config-driven entry point for a discrete map system -- the
 * `MapType` counterpart to run_ode_generic(). Same overall behavior (see
 * run_ode_generic()'s docs), with map-shaped config keys:
 *  - `calculation`: one of `escape`, `msd`, `lyapunov`, `phasespace`.
 *  - `output_file`, `grid_dims`, `grid_min`, `grid_max`: same as run_ode_generic().
 *  - `iterations` (all calculation types): number of map iterations to run
 *    (there's no `dt`/`final_time` for a discrete map).
 *  - Any key the system's own `load_params_for<ParamsType>` reads (see
 *    that system's header, e.g. maps/standard_map.h).
 *
 * Unlike the ODE runner, `msd` here is **not** log-sampled or batched (one
 * MSD sample per iteration, one kernel launch) -- fine at the iteration
 * counts map calculations typically need; see ode_msd.cuh's docs for why
 * the ODE side needed both. `phasespace` is CPU-only (see
 * calculate_phase_space() in cuda_dynamics.h) -- slow at large grid/iteration counts.
 *
 * HDF5 datasets written (see run_ode_generic() for the sweep-suffix
 * convention): `EscapeTime`/`EscapeBasin` (escape); `MSD`/`Displacement`/
 * `TotalDisplacement` (msd, no shared sample-times dataset since sampling
 * is dense); `Lyapunov` (lyapunov); `PhaseSpace` (phasespace).
 *
 * @tparam DIMS State dimensionality (fixed at compile time; see run_ode_generic()).
 * @tparam MapType The map functor type, e.g. `StandardMap`.
 * @tparam ParamsType That map's parameter struct type.
 * @param argc,argv Command-line arguments as passed to `main()`; expects
 * exactly one argument, the config file path.
 * @return `0` on success; `1` on any error, after printing a message to stderr.
 */
template <int DIMS, typename MapType, typename ParamsType>
inline int run_map_generic(int argc, char** argv) {
    using namespace cudat_runner_detail;
    if (argc < 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "run") << " <config_file>" << std::endl;
        return 1;
    }

    try {
        Config base_cfg = Config::load(argv[1]);

        const std::string calculation = base_cfg.get_string("calculation");
        const std::string output_file = base_cfg.get_string("output_file");

        std::vector<int> grid_res = require_dims_list(base_cfg, "grid_dims", DIMS);
        std::vector<double> grid_min = require_dims_list_d(base_cfg, "grid_min", DIMS);
        std::vector<double> grid_max = require_dims_list_d(base_cfg, "grid_max", DIMS);
        GridSetup grid(DIMS, grid_res, grid_min, grid_max);

        MapType map;
        prepare_output_file(output_file);

        std::vector<SweepResult> sweeps = enumerate_sweeps(base_cfg);
        std::cout << "run_map_generic: " << sweeps.size() << " run(s), calculation='"
                  << calculation << "', " << grid.num_particles << " particles" << std::endl;

        for (size_t i = 0; i < sweeps.size(); ++i) {
            const Config& cfg = sweeps[i].config;
            const std::string& suffix = sweeps[i].suffix;
            const std::string dset_suffix = suffix.empty() ? "" : ("_" + suffix);
            ParamsType params = load_params_for<ParamsType>(cfg);

            if (!suffix.empty()) {
                std::cout << "\n--- [" << (i + 1) << "/" << sweeps.size() << "] " << suffix << " ---" << std::endl;
            }

            if (calculation == "escape") {
                const int iterations = cfg.get_int("iterations");

                std::vector<double> h_escape_times(grid.num_particles);
                std::vector<double> h_escape_basins(grid.num_particles);
                calculate_escape_time<DIMS, MapType, ParamsType>(
                    map, params, grid.h_initial_conditions.data(), grid.num_particles,
                    iterations, h_escape_times.data(), h_escape_basins.data());

                std::vector<size_t> dims(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "EscapeTime" + dset_suffix, dims, h_escape_times.data());
                save_to_h5(output_file, "EscapeBasin" + dset_suffix, dims, h_escape_basins.data());

            } else if (calculation == "msd") {
                const int iterations = cfg.get_int("iterations");

                std::vector<double> h_total_disp(grid.num_particles);
                std::vector<double> h_disp(grid.num_particles * DIMS);
                std::vector<double> h_msd(iterations); // one sample per iteration (no log-sampling for maps yet)
                calculate_msd_and_displacement<DIMS, MapType, ParamsType>(
                    map, params, grid.h_initial_conditions.data(), nullptr, nullptr,
                    grid.num_particles, iterations,
                    h_total_disp.data(), h_disp.data(), h_msd.data());

                std::vector<size_t> msd_dims = { static_cast<size_t>(iterations) };
                std::vector<size_t> grid_dims_2d(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "MSD" + dset_suffix, msd_dims, h_msd.data());
                save_displacement_components(output_file, "Displacement" + dset_suffix, grid, h_disp.data());
                save_to_h5(output_file, "TotalDisplacement" + dset_suffix, grid_dims_2d, h_total_disp.data());

            } else if (calculation == "lyapunov") {
                const int iterations = cfg.get_int("iterations");

                std::vector<double> h_lyap(grid.num_particles);
                calculate_lyapunov_exponent<DIMS, MapType, ParamsType>(
                    map, params, grid.h_initial_conditions.data(), grid.num_particles,
                    iterations, h_lyap.data());

                std::vector<size_t> dims(grid.grid_res.begin(), grid.grid_res.end());
                save_to_h5(output_file, "Lyapunov" + dset_suffix, dims, h_lyap.data());

            } else if (calculation == "phasespace") {
                // calculate_phase_space is CPU-only (no GPU kernel exists for it);
                // fine for small grids/iteration counts, slow for large ones.
                const int iterations = cfg.get_int("iterations");

                std::vector<double> h_phase(static_cast<size_t>(grid.num_particles) * iterations * DIMS);
                calculate_phase_space<DIMS, MapType, ParamsType>(
                    map, params, grid.h_initial_conditions.data(), grid.num_particles,
                    iterations, h_phase.data());

                std::vector<size_t> dims = { static_cast<size_t>(grid.num_particles),
                                              static_cast<size_t>(iterations),
                                              static_cast<size_t>(DIMS) };
                save_to_h5(output_file, "PhaseSpace" + dset_suffix, dims, h_phase.data());

            } else {
                std::cerr << "Unknown calculation '" << calculation
                          << "' (expected escape|msd|lyapunov|phasespace)" << std::endl;
                return 1;
            }

            std::cout << "Saved '" << calculation << "' results"
                      << (suffix.empty() ? "" : (" for " + suffix)) << "." << std::endl;
        }

        std::cout << "\nAll '" << calculation << "' runs complete." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
