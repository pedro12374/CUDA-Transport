import plotting_lib as pl
import numpy as np # Needed for np.pi
import parana_theme as tema

# ==============================================================================
# == DEFINE ALL PLOTS TO BE GENERATED
# ==============================================================================

individual_matrix_batches = [
    {
        "h5_file": "../dat/horton_escape_A2_A3.h5",
        "output_dir": "../plots", # Directory to save files
        "plot_type": "escape",
        "dset_prefix": "EscapeTime",
        "row_params": [0.0,0.005, 0.013],      # A3 values
        "col_params": [0.0,0.005, 0.013, 0.026], # A2 values
        "row_prefix": "A3",
        "col_prefix": "A2",
        "bounds": [-np.pi, np.pi, -2*np.pi, 2*np.pi],
        "basin_cmap_config": {
            "colors": [
                tema.VERMELHO,      # Color for basin -2 (e.g., escape left)
                tema.BACKGROUND_COLOR, # Color for basin 0 (no escape)
                tema.AMARELO        # Color for basin 2 (e.g., escape right)
            ],
            "bounds": [ -1.5, -0.5, 0.5, 1.5],
            "ticks": [ -1, 0, 1]
        }
    },
        {
        "h5_file": "../dat/horton_escape_A2_A3.h5",
        "output_dir": "../plots", # Directory to save files
        "plot_type": "basin",
        "dset_prefix": "EscapeBasin",
        "row_params": [0.0,0.005, 0.013],      # A3 values
        "col_params": [0.0,0.005, 0.013, 0.026], # A2 values
        "row_prefix": "A3",
        "col_prefix": "A2",
        "bounds": [-np.pi, np.pi, -2*np.pi, 2*np.pi],
        "basin_cmap_config": {
            "colors": [
                tema.VERMELHO,      # Color for basin -2 (e.g., escape left)
                tema.BACKGROUND_COLOR, # Color for basin 0 (no escape)
                tema.AMARELO        # Color for basin 2 (e.g., escape right)
            ],
            "bounds": [ -1.5, -0.5, 0.5, 1.5],
            "ticks": [ -1, 0, 1]
        }
    },

    {
        "h5_file": "../dat/horton_msd_A2_A3.h5",
        "output_dir": "../plots", # Directory to save files
        "plot_type": "msd",
        "dset_prefix": "MSD",
        "row_params": [0.0,0.005, 0.013],      # A3 values
        "col_params": [0.0,0.005, 0.013, 0.026], # A2 values
        "row_prefix": "A3",
        "col_prefix": "A2",
        "bounds": [-np.pi, np.pi, -2*np.pi, 2*np.pi]
    },

    {
        "h5_file": "../dat/horton_msd_A2_A3.h5",
        "output_dir": "../plots", # Directory to save files
        "plot_type": "total_displacement",
        "dset_prefix": "TotalDisplacement",
        "row_params": [0.0,0.005, 0.013],      # A3 values
        "col_params": [0.0,0.005, 0.013, 0.026], # A2 values
        "row_prefix": "A3",
        "col_prefix": "A2",
        "bounds": [-np.pi, np.pi, -2*np.pi, 2*np.pi]
    }    
]


# ==============================================================================
# == EXECUTION LOOP
# ==============================================================================

if __name__ == '__main__':
    if individual_matrix_batches:
        print("\n--- Generating Individual Plots from 2D Sweep ---")
        for config in individual_matrix_batches:
            
            pl.generate_individual_matrix_plots(**config)

    print("\nAll plots have been generated.")