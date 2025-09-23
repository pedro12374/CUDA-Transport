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
OUTPUT_PDF = '../plots/standard_map_MMLE.pdf'

df = pd.DataFrame()
for k_val in K_VALUES:
    with h5py.File(H5_FILENAME, 'r') as f:
        dset_name = f"K_{k_val:.6f}"
        data = f[dset_name][:]
    data = pd.DataFrame(np.reshape(data, (-1))).transpose().melt(var_name='Part', value_name=f'K_{k_val:.6f}')
    df = pd.concat([df, data], axis=1)

df_final = df.drop(columns=['Part'])

fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

ax.plot(df_final.mean())

ax.set_xticks(ticks=range(len(K_VALUES)), labels=K_VALUES)

ax.set_xlabel(r'$K$')
ax.set_ylabel(r'$\Lambda$')




plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')