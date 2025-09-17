import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm 
# --- Configuration ---
# Choose which K value you want to visualize
K_VALUE_TO_ANALYZE = 6.28

# Define the phase space boundaries from your C++ code
P_MIN = -np.pi
P_MAX = np.pi
THETA_MIN = 0.0
THETA_MAX = 2.0 * np.pi

# --- Analysis & Visualization ---

# The dataset name inside the HDF5 file is based on the K value
# NEW, CORRECTED LINE ✅
dset_name = f"K_{K_VALUE_TO_ANALYZE:.6f}"

print(f"Loading and plotting the displacement map for dataset: '{dset_name}'")

try:
    # Use a 'with' block to automatically open and close the file
    with h5py.File('../dat/displacement_p.h5', 'r') as f:
        # Check if the requested dataset exists in the file
        if dset_name not in f:
            raise KeyError(f"Dataset '{dset_name}' not found in total_displacement.h5. "
                           f"Available datasets are: {list(f.keys())}")
            
        # Load the 2D displacement map into a numpy array
        displacement_map = f[dset_name][:]

    # --- Plotting the 2D Map ---
    plt.figure(figsize=(8, 7))
    limit = np.max(np.abs(displacement_map))
    
    # 3. Create the TwoSlopeNorm object, centered at 0.
    # It will map data from -limit to +limit.
    centered_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    # We use imshow to create a color plot of the 512x512 matrix.
    # 'extent' sets the axis labels to the physical coordinates of phase space.
    # 'origin=lower' puts the (0,0) point at the bottom-left corner.
    im = plt.imshow(displacement_map.T, 
                    origin='lower', 
                    extent=[THETA_MIN, THETA_MAX, P_MIN, P_MAX],
                    aspect='auto',
                    cmap='bwr',
                    norm=centered_norm)

    plt.colorbar(im, label='Total Final Displacement')
    plt.title(f'Final Displacement Map for K = {K_VALUE_TO_ANALYZE}')
    plt.xlabel(r'Initial Angle $\theta_0$')
    plt.ylabel(r'Initial Momentum $p_0$')
    
    # Save the figure to a file
    output_filename = f"displacement_map_K_{K_VALUE_TO_ANALYZE}.png"
    plt.savefig(output_filename)
    print(f"Saved displacement map to {output_filename}")
    
    # Display the plot on the screen
    plt.show()

except (FileNotFoundError, KeyError) as e:
    print(f"An error occurred: {e}")