import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # Use logarithmic scale for better contrast
import parana_theme as tema
tema.aplicar_tema()

# --- Configuration ---
K_VALUES = [0.5, 0.971635, 1.5, 6.47] 
H5_FILENAME = '../dat/Escape.h5'
OUTPUT_PDF = '../plots/standard_map_Escape_Basins.pdf'

# Phase space boundaries from the simulation
P_MIN, P_MAX = -np.pi,  np.pi
THETA_MIN, THETA_MAX = 0.0, 2.0 * np.pi

cmap, norm = tema.get_escape_basin_cmap_norm()

# --- Plotting ---
mosaic = [['a', 'b'], 
          ['c', 'd']]
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained')
fig.set_size_inches(10, 8)  
print(f"Generating Lyapunov exponent plot for K = {K_VALUES}...")

for k_val, label in zip(K_VALUES, axs.keys()):
    ax = axs[label]
    dset_name = f"Basin_K_{k_val:.6f}"
    
    try:
        with h5py.File(H5_FILENAME, 'r') as f:
            if dset_name not in f:
                print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
                continue
            map = f[dset_name][:]
        if(np.isin(1,map)):
            print("Found 1 in map")
        # Use a logarithmic color scale to highlight the differences
        # We add a small epsilon to avoid taking the log of zero for stable orbits
        im = ax.imshow(map, 
                       origin='lower', 
                       extent=[THETA_MIN, THETA_MAX, P_MIN, P_MAX],
                       aspect='auto',
                       cmap=cmap,norm=norm # 'inferno' or 'plasma' are good for this
                       )
        
        cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
        cbar.set_label(r'Escape Basin')

        ax.text(0.05, 0.95, f'{label})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', color=tema.TEXT_COLOR,
                bbox=dict(facecolor=tema.BACKGROUND_COLOR, alpha=0.7, edgecolor='none', pad=2.0))
        title_k = r'$K_c \approx 0.97$' if np.isclose(k_val, 0.971635) else f'K = {k_val}'

        title_k = r'$K_c \approx 0.97$' if np.isclose(k_val, 0.971635) else f'K = {k_val}'
        ax.set_title(title_k)

    except (FileNotFoundError, KeyError) as e:
        ax.text(0.5, 0.5, f"Error loading\n{dset_name}", ha='center', va='center')
        print(f"Error for K={k_val}: {e}")

for label in ['a', 'c']:
    axs[label].set_ylabel(r'$p_0$')
for label in ['c', 'd']:
    axs[label].set_xlabel(r'$\theta_0$')

plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
print(f"Saved plot to {OUTPUT_PDF}")
