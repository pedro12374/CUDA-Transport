import h5py
import numpy as np
import matplotlib.pyplot as plt
import parana_theme as pt
pt.aplicar_tema()
# --- Configuration ---
K_VALUES = [0.5, 0.971635, 1.5, 6.47] 
H5_FILENAME = '../dat/PS_Zoom.h5'
OUTPUT_PDF = '../plots/standard_map_phasespace.pdf'

# Phase space boundaries
P_MIN, P_MAX = -np.pi,  np.pi
THETA_MIN, THETA_MAX = 0.0, 2*np.pi

# --- Plotting ---


fig, ax = plt.subplots()
fig.set_size_inches(4, 4)  
print(f"Generating Lyapunov exponent plot for K = {K_VALUES}...")

k_val = K_VALUES[0]  # Change index to select different K
dset_name = f"K_{k_val:.6f}"
with h5py.File(H5_FILENAME, 'r') as f:
            if dset_name not in f:
                print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
            trajectories = f[dset_name][:]
all_p_vals = trajectories[:, :, 0].flatten()
all_theta_vals = trajectories[:, :, 1].flatten()


all_p_vals = (all_p_vals + P_MAX) % (2 * P_MAX) - P_MAX
all_theta_vals = all_theta_vals % THETA_MAX

ax.plot(all_theta_vals, all_p_vals, ',', color='black')

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p$')

ax.set_xlim(0,0.5)
ax.set_ylim(-0.0, 0.2)

plt.savefig('standard_map_phasespace_Zoom.png', bbox_inches='tight',dpi=300)
