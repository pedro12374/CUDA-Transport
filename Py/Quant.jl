# Filename: basin_entropy.jl

using HDF5
using Printf
using DelimitedFiles
using Attractors # Using the robust, standard library for the calculation
using Base.Threads

# ==============================================================================
# == HELPER FUNCTION
# ==============================================================================

"""
    read_basin_data(h5_file, dset_name, grid_dim)

Reads a basin dataset created by the custom C++ code and reshapes it.
"""
function read_basin_data(h5_file, dset_name, grid_dim)
    # Read the 1D array of basin data
    basin_1d = h5read(h5_file, dset_name)
    
    # Reshape the 1D data into a 2D matrix using the provided grid dimension
    # The Attractors.jl functions expect integer matrices
    return Int.(reshape(basin_1d, (grid_dim, grid_dim)))
end

# ==============================================================================
# == MAIN SCRIPT
# ==============================================================================

function main()
    # --- 1. Configuration ---
    K_VALUES = [0.5, 0.971635, 1.5, 6.47]
    H5_FILENAME = "../dat/Escape.h5"
    OUTPUT_FILE = "../dat/basin_entropy_results_box4.dat"
    
    # Set the fixed box size for the calculation
    BOX_SIZE = 4
    
    # IMPORTANT: This must match the `grid_dim` used to generate the .h5 file
    GRID_DIM = 1024 

    println("Starting basin entropy calculation for a fixed box size.")
    println("K values: $(K_VALUES)")
    println("Box size: $(BOX_SIZE)")

    # --- 2. Data Structures ---
    num_k_values = length(K_VALUES)
    s_bb_results = zeros(num_k_values) # Simple array for single box size
    s_b_results = zeros(num_k_values)
    A_neg1 = zeros(num_k_values)
    A_0 = zeros(num_k_values)
    A_1 = zeros(num_k_values)

    # --- 3. Main Calculation Loop ---
    @threads for i in 1:num_k_values
        k_val = K_VALUES[i]
        dset_name = @sprintf("Basin_K_%.6f", k_val)
        
        try
            bacia = read_basin_data(H5_FILENAME, dset_name, GRID_DIM)
            
            # Calculate basin fractions
            fractions_dict = basins_fractions(bacia)
            A_neg1[i] = get(fractions_dict, -1, 0.0)
            A_0[i] = get(fractions_dict, 0, 0.0)
            A_1[i] = get(fractions_dict, 1, 0.0)
            
            # Calculate entropy for the single, fixed box size
            s_b_results[i], s_bb_results[i] = basin_entropy(bacia, BOX_SIZE)
            
        catch e
            println("Warning: Could not process dataset for K=$(k_val). Error: $e")
        end
    end
    println("Calculations finished.")

    # --- 4. Save Results ---
    # Simplified header for a single box size
    header = ["K", "A_neg1", "A_0", "A_1","S_b", "S_bb"]
    
    # Combine all results into a single matrix
    final_data = hcat(K_VALUES, A_neg1, A_0, A_1,s_b_results, s_bb_results)

    open(OUTPUT_FILE, "w") do io
        writedlm(io, [join(header, "\t")])
        writedlm(io, final_data, '\t')
    end
    
    println("✅ Results successfully saved to: $(OUTPUT_FILE)")
end

# Run the main function
main()