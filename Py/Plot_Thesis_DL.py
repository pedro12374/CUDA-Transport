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
H5_FILENAME = '../dat/lyapunov_exponents.h5'
OUTPUT_PDF = '../plots/standard_map_MLED.pdf'

df = pd.DataFrame()
for k_val in K_VALUES:
    with h5py.File(H5_FILENAME, 'r') as f:
        dset_name = f"K_{k_val:.6f}"
        data = f[dset_name][:]
    data = pd.DataFrame(np.reshape(data, (-1))).transpose().melt(var_name='Part', value_name=f'K_{k_val:.6f}')
    df = pd.concat([df, data], axis=1)


fig, ax = plt.subplot_mosaic([['A'],[ 'B'], ['C'],[ 'D']], figsize=(6, 6), layout='constrained',sharex=True)
sns.kdeplot(data=df["K_0.500000"], ax=ax["A"])
sns.kdeplot(data=df["K_0.971635"], ax=ax["B"])
sns.kdeplot(data=df["K_1.500000"], ax=ax["C"])
sns.kdeplot(data=df["K_6.470000"], ax=ax["D"])



ax['A'].set_title(r'$K=0.5$',y=0.8)
ax['B'].set_title(r'$K=0.971635$',y=0.8)
ax['C'].set_title(r'$K=1.5$',y=0.8)
ax['D'].set_title(r'$K=6.47$',y=0.8)

ax['A'].set_xlabel(r'$\lambda$')
ax['B'].set_xlabel(r'$\lambda$')
ax['C'].set_xlabel(r'$\lambda$')
ax['D'].set_xlabel(r'$\lambda$')





plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')