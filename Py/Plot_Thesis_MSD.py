import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import TwoSlopeNorm 
from scipy.optimize import curve_fit
import parana_theme as tema
tema.aplicar_tema()
# --- Configuration ---
# List of K values to plot in the mosaic
K_VALUES = [0.5,  1.5, 6.47] 
H5_FILENAME = '../dat/msd_p.h5'
OUTPUT_PDF = '../plots/standard_map_MSD.pdf'



def power_law(t, D, alpha):
    """Power-law model for MSD: MSD = D * t^alpha"""
    return D * (t**alpha)

fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

for k_val in K_VALUES:
    dset_name = f"K_{k_val:.6f}"
    

    with h5py.File(H5_FILENAME, 'r') as f:
        if dset_name not in f:
            print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
            continue
        
        msd_data = f[dset_name][:]
        iterations = np.arange(len(msd_data))

    popt, pcov = curve_fit(power_law, iterations, msd_data)

    D = popt[0]
    alpha = popt[1]

    errors = np.sqrt(np.diag(pcov))
    D_error = errors[0]
    alpha_error = errors[1]

    fit_line = power_law(iterations, D, alpha)
    # 3. Plot the 2D map on the current subplot axis
    ax.loglog(iterations[1:], msd_data[1:], 'o', markersize=3,color=tema.parana_colors[K_VALUES.index(k_val)])
    ax.loglog(iterations, fit_line, '-', linewidth=1, label=fr'$K={k_val}$ - $\alpha$={alpha:.0f}',color=tema.parana_colors[K_VALUES.index(k_val)])
    ax.legend()
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\langle\sigma(t)^2\rangle $')


plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')