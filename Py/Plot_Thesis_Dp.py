import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import TwoSlopeNorm 
import parana_theme as tema
tema.aplicar_tema()
# --- Configuration ---
# List of K values to plot in the mosaic
K_VALUES = [0.5, 0.971635, 1.5, 6.47] 
H5_FILENAME = '../dat/displacement_components.h5'
OUTPUT_PDF = '../plots/standard_map_displacement.pdf'

# Phase space boundaries from the simulation
P_MIN, P_MAX = -np.pi,  np.pi
THETA_MIN, THETA_MAX = 0.0, 2.0 * np.pi



# 1. Set up the figure and the mosaic layout
# This creates a 2x2 grid where each subplot is labeled 'a', 'b', 'c', or 'd'.
mosaic = [['a', 'b'], 
          ['c', 'd']]

fig, axs = plt.subplot_mosaic(mosaic,layout='constrained', gridspec_kw={
        "wspace": -0.1,
        "hspace": -0.1,
    },)
fig.set_size_inches(10, 8)  
print(f"Generating mosaic plot for K = {K_VALUES}...")

# 2. Loop through the K values and subplot labels to populate the figure
for k_val, label in zip(K_VALUES, axs.keys()):
    ax = axs[label] # Get the specific subplot axis to draw on
    dset_name = f"K_{k_val:.6f}"
    
    try:
        with h5py.File(H5_FILENAME, 'r') as f:
            if dset_name not in f:
                print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
                continue
            full_displacement_data = f[dset_name][:]
            # CORRECTED: Select only the momentum component (index 0) for plotting
            displacement_map = full_displacement_data[:, :, 0]


        limit = np.max(np.abs(displacement_map))
    

        centered_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        # 3. Plot the 2D map on the current subplot axis
        im = ax.imshow(displacement_map, 
                       origin='lower', 
                       extent=[THETA_MIN, THETA_MAX, P_MIN, P_MAX],
                       aspect='auto',
                       cmap=tema.parana_div_yel_blu,
                       norm=centered_norm)
        
        # Add a colorbar specific to this subplot
        fig.colorbar(im, ax=ax, label=r'$\Delta p$')

        # Add the a), b), c) label to the corner of the subplot
        ax.text(0.05, 0.95, f'{label})', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', color='black')

        # Set a title for each subplot
        # If K is the critical value, give it a special label
        title_k = r'$K_c \approx 0.97$' if np.isclose(k_val, 0.971635) else f'K = {k_val}'
        ax.set_title(title_k)

    except (FileNotFoundError, KeyError) as e:
        ax.text(0.5, 0.5, f"Error loading\n{dset_name}", ha='center', va='center')
        print(f"Error for K={k_val}: {e}")

# Add axis labels only to the outer plots to keep the figure clean
for label in ['a', 'c']:
    axs[label].set_ylabel(r'$p_0$')
for label in ['c', 'd']:
    axs[label].set_xlabel(r'$\theta_0$')




# 4. Save the completed figure to a PDF file
#    bbox_inches='tight' trims whitespace, which is great for LaTeX.
plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')

