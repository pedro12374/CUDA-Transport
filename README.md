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

- [ ] **Implement Configuration Files:** Instead of hardcoding simulation parameters (A2/A3 values, grid size, number of iterations) in the `main` files, move them to an external configuration file (e.g., `config.json` or `params.txt`).
    - [ ] Modify the `main` executables to read these configuration files at startup.
- [ ] **Migrate the Build System to CMake (Advanced):** To facilitate compilation across different systems, replace the `Makefile` with `CMake`.
    - [ ] `CMake` can automatically find dependencies (HDF5, etc.), which makes compilation much easier for other users.

### Medium Priority: Robustness & Testing

Ensure that the results are always correct and the code is reliable.

- [ ] **Create a Test Suite:**
    - [ ] Add a `tests/` directory.
    - [ ] Write simple tests that verify the solvers produce known results for simple cases (e.g., verify that a stable orbit in the Standard Map for a low K value remains confined).
    - [ ] This ensures that future code changes do not accidentally break the physics of the calculations.

### Low Priority: Refactoring & Features

Finalize the code structure and add new functionality.

- [ ] **Finalize the "Header-Only" Refactor:** Ensure all solvers (`map_escape`, `map_lyapunov`, etc.) have been moved to their own `.cuh` files and that the old library `.cu` files have been removed.
- [ ] **Unify Python Plotting Scripts:** Consolidate all old `Plot_Thesis_*.py` scripts into the new structure with `plotting_lib.py` and `run_plots.py` to avoid code duplication.