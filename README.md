# CUDA-Transport

Codes to calculate transport and chaotic analyzs using CUDA 


# To-Do List & Development Roadmap

This is a list of planned improvements to evolve `CUDA-Transport` from a personal project into a robust, reusable scientific library.

### High Priority: Documentation

Documentation is the most critical step to make the library usable by others.

- [ ] **Write a Comprehensive `README.md`:** The current README is almost empty. It needs to include:
    - [ ] A clear description of what the `CUDA-Transport` library is and what problems it solves.
    - [ ] A list of **dependencies** and how to install them (CUDA Toolkit, HDF5, HighFive).
    - [ ] Detailed **build instructions** (explaining how to configure the `Makefile` or a future `CMake`).
    - [ ] A **Quick Start Guide** showing a complete example of how to run a simulation and generate a plot.
- [ ] **Add Doxygen-Style Comments to the C++ Code:** Comment the code using a standard format like **Doxygen**.
    - [ ] Add comments to the headers (`.h`, `.cuh`) explaining what each function does, its parameters, and what it returns.
    - [ ] Explain the required interface for adding new dynamical systems (the structure of the `operator()` and `jacobian` methods).
- [ ] **Create a Tutorial:** Write a `TUTORIAL.md` file that guides a new user through the process of:
    - [ ] Defining a new dynamical system in a header file.
    - [ ] Creating a `main` file to run an analysis (e.g., `escape_time`) on this new system.
    - [ ] Generating a plot from the results.

### Medium Priority: Usability & Configuration

Improve how simulations are configured and executed.

- [x] **Implement Configuration Files:** Instead of hardcoding simulation parameters (A2/A3 values, grid size, number of iterations) in the `main` files, move them to an external configuration file (e.g., `config.json` or `params.txt`).
    - [x] Modify the `main` executables to read these configuration files at startup.
    - Done via a generic, config-driven runner rather than per-driver parsing: see `cuda_dynamics_lib/include/config.h`/`runner.cuh` and `configs/*.cfg`. A dependency-free `key = value` format was used instead of JSON (no JSON dev headers installed on the dev server, no passwordless sudo to add one).
- [x] **Migrate the Build System to CMake (Advanced):** To facilitate compilation across different systems, replace the `Makefile` with `CMake`.
    - [x] `CMake` can automatically find dependencies (HDF5, etc.), which makes compilation much easier for other users.
    - Added (`CMakeLists.txt`) as an alternative to the Makefile, per the original phase plan ("keep the Makefile working until CMake is verified") -- both build systems work and are kept in sync. `find_package(HDF5 COMPONENTS CXX)` finds it automatically via pkg-config, no manual path hints needed on this server. `mkdir build && cd build && cmake .. && make -j && ctest`.

### Medium Priority: Robustness & Testing

Ensure that the results are always correct and the code is reliable.

- [x] **Create a Test Suite:**
    - [x] Add a `tests/` directory.
    - [x] Write simple tests that verify the solvers produce known results for simple cases (e.g., verify that a stable orbit in the Standard Map for a low K value remains confined).
    - [x] This ensures that future code changes do not accidentally break the physics of the calculations.
    - Run with `make test`; see `tests/README.md` for what's covered (GPU-vs-CPU reference trajectories, RK4 convergence order, confined regular orbits, conserved quantities, batching regression).

### Low Priority: Refactoring & Features

Finalize the code structure and add new functionality.

- [x] **Finalize the "Header-Only" Refactor:** Ensure all solvers (`map_escape`, `map_lyapunov`, etc.) have been moved to their own `.cuh` files and that the old library `.cu` files have been removed.
- [x] **Unify Python Plotting Scripts:** Consolidate all old `Plot_Thesis_*.py` scripts into the new structure with `plotting_lib.py` and `run_plots.py` to avoid code duplication.
    - No `Plot_Thesis_*.py` files remained by the time this was picked up. Found and fixed instead: `Presentation/parana_theme.py` was a byte-identical duplicate of `Py/parana_theme.py` (removed, `Presentation/plt.py` now imports the canonical copy), and `Presentation/plt.py` itself was reading a file (`PS_Zoom.h5`) no driver produces anymore (its source, `main_PS.cu.bkp`, was removed earlier) -- restored via `configs/standard_map_phasespace_zoom.cfg` against the current generic pipeline.