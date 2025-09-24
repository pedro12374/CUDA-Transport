import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import TwoSlopeNorm 
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
import parana_theme as tema
tema.aplicar_tema()
# --- Configuration ---
# List of K values to plot in the mosaic
K_VALUES = [0.5, 0.971635, 1.5, 6.47] 
H5_FILENAME = '../dat/displacement_components.h5'
OUTPUT_PDF = '../plots/standard_map_dist.pdf'

df = pd.DataFrame()
for k_val in K_VALUES:
    with h5py.File(H5_FILENAME, 'r') as f:
        dset_name = f"K_{k_val:.6f}"
        full_displacement_data = f[dset_name][:]
        data = full_displacement_data[:, :, 0]
    data = pd.DataFrame(np.reshape(data, (-1))).transpose().melt(var_name='Part', value_name=f'K_{k_val:.6f}')
    df = pd.concat([df, data], axis=1)


mosaic = [['a', 'b'], 
          ['c', 'd']]

fig, axs = plt.subplot_mosaic(mosaic,layout='constrained', gridspec_kw={
        "wspace": -0.1,
        "hspace": -0.1,
    },)
fig.set_size_inches(10, 8)  


for k_val, label in zip(K_VALUES, axs.keys()):
    ax = axs[label] # Get the specific subplot axis to draw on
    dset_name = f"K_{k_val:.6f}"
    sns.histplot(data=df[dset_name], ax=ax,bins=50)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta p$')
    title_k = r'$K_c \approx 0.97$' if np.isclose(k_val, 0.971635) else f'K = {k_val}'
    ax.set_title(title_k)
    ax.text(0.05, 0.95, f'{label})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', color=tema.TEXT_COLOR,
            bbox=dict(facecolor=tema.BACKGROUND_COLOR, alpha=0.7, edgecolor='none', pad=2.0))




plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')