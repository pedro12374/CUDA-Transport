# File: Py/plot_entropy.py

import pandas as pd
import matplotlib.pyplot as plt
import parana_theme as tema

# Apply your custom theme
tema.aplicar_tema()

# --- 1. Configuration ---
INPUT_FILE = '../dat/basin_entropy_results_box4.dat'
OUTPUT_PDF = '../plots/basin_entropy_vs_K.pdf'
BOX_SIZE = 4 # The box size used in the Julia script

# --- 2. Load Data ---
try:
    # Use pandas to easily read the tab-delimited file
    df = pd.read_csv(INPUT_FILE, sep='\t')
except FileNotFoundError:
    print(f"Error: Input file not found at '{INPUT_FILE}'")
    exit()

# --- 3. Create the Plot ---
fig, ax1 = plt.subplots(figsize=(6, 3))

# Create the second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# --- 4. Plot Basin Fractions on the Left Axis (ax1) ---
# Use the theme's colors for plotting

ax1.set_xlabel('K ')
ax1.set_ylabel('Entropy')
ax1.set_ylim(0, 0.7)

# Plot the fraction of non-escaping orbits
p1, = ax1.plot(df['K'], df['S_b'], color=tema.AZUL, linestyle='--', marker='o', label=r'$S_b$')
# Plot the fractions of escaping orbits (sum of positive and negative)
p2, = ax1.plot(df['K'], df['S_bb'], color=tema.VERDE, linestyle='--', marker='^', label=r'$S_{bb}$')

# --- 5. Plot Basin Entropy on the Right Axis (ax2) ---

ax2.set_ylabel(f'Area')
ax2.set_ylim(0,1)

# Plot the two entropy measures
p3, = ax2.plot(df['K'], df[f'A_1'], color=tema.AMARELO, linestyle='-', marker='s', label=r'$Area_{\pi}$')
p4, = ax2.plot(df['K'], df[f'A_neg1'], color=tema.VERMELHO, linestyle='--', marker='o', label=r'$Area_{-\pi}$')

# --- 6. Final Touches ---
fig.tight_layout()

# Create a single, unified legend for all lines
lines = [p1, p2, p3, p4]
fig.legend(lines, [l.get_label() for l in lines], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
plt.subplots_adjust(top=0.9) # Adjust top to make space for the legend

# --- 7. Save the Figure ---
plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
print(f"✅ Plot successfully saved to: {OUTPUT_PDF}")

# To display the plot interactively (optional)
# plt.show()