import h5py
import numpy as np
import matplotlib.pyplot as plt
import parana_theme as tema
tema.aplicar_tema()

# --- Configuration ---
K_VALUES = [0.5, 0.971635, 1.5, 6.47] 
H5_FILENAME = '../dat/phase_space.h5'
OUTPUT_PDF = '../plots/standard_map_phasespace.pdf'

# Phase space boundaries
P_MIN, P_MAX = -np.pi,  np.pi
THETA_MIN, THETA_MAX = 0.0, 2.0 * np.pi

# --- Plotting ---

mosaic = [['a', 'b'], 
          ['c', 'd']]
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained')
fig.set_size_inches(8, 8)  
print(f"Generating Lyapunov exponent plot for K = {K_VALUES}...")

for k_val, label in zip(K_VALUES, axs.keys()):
    ax = axs[label]
    dset_name = f"K_{k_val:.6f}"
    
    try:
        with h5py.File(H5_FILENAME, 'r') as f:
            if dset_name not in f:
                print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
                continue
            trajectories = f[dset_name][:]
        
        print(f"Plotting {trajectories.shape[0]} orbits for K={k_val}...")
        
        all_p_vals = trajectories[:, :, 0].flatten()
        all_theta_vals = trajectories[:, :, 1].flatten()

        # --- WRAPPING LOGIC ---
        # Apply wrapping to all points at once for efficiency
        all_p_vals = (all_p_vals + P_MAX) % (2 * P_MAX) - P_MAX
        all_theta_vals = all_theta_vals % THETA_MAX
        
        # Plotting with small, semi-transparent dots gives the classic phase space look
            
            # Plotting with small, semi-transparent dots gives the classic phase space look
        ax.plot(all_theta_vals, all_p_vals, ',', color='black', markersize=0.2, alpha=0.6, rasterized=True)

        # --- Formatting ---
        title_k = r'$K_c \approx 0.97$' if np.isclose(k_val, 0.971635) else f'K = {k_val}'
        ax.set_title(title_k)
        ax.set_xlim(THETA_MIN, THETA_MAX)
        ax.set_ylim(P_MIN, P_MAX)

    except (FileNotFoundError, KeyError) as e:
        ax.text(0.5, 0.5, f"Error loading\n{dset_name}", ha='center', va='center')
        print(f"Error for K={k_val}: {e}")

for label in ['a', 'c']:
    axs[label].set_ylabel(r'$p_0$')
for label in ['c', 'd']:
    axs[label].set_xlabel(r'$\theta_0$')

plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
print(f"Saved plot to {OUTPUT_PDF}")