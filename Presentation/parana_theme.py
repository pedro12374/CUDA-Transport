import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from matplotlib import cycler
import numpy as np

# 1. Define the final, curated color palette
parana_colors = [
    '#009639', # Verde Bandeira
    '#b30000', # Vermelho Terra
    '#0052a5', # Azul Royal
    '#e6007e', # Plasma Magenta
    '#6a00ff', # Azul Violeta
    '#ffd700', # Ouro Ipê
]

VERDE, VERMELHO, AZUL, MAGENTA, VIOLETA, AMARELO = parana_colors

# 2. Define custom colormaps

# --- Original Colormaps ---
# Sequential: Light Yellow -> Verde Bandeira
parana_seq = LinearSegmentedColormap.from_list(
    'parana_seq',
    ['#ffffcc', parana_colors[0]]
)

# Diverging: Vermelho Terra -> Light Gray -> Azul Royal
parana_div = LinearSegmentedColormap.from_list(
    'parana_div',
    [parana_colors[1], '#f8f8f8', parana_colors[2]]
)


# --- ✨ NEW: Richer, Multi-Segment Colormaps ---

# Rich Sequential: Light Yellow -> Ouro Ipê -> Verde Bandeira
parana_seq_rich = LinearSegmentedColormap.from_list(
    'parana_seq_rich',
    ['#ffffcc', parana_colors[5], parana_colors[0]]
)

# --- NEW DIVERGING OPTION 1: Violet -> Gray -> Green ---
# This is an excellent choice for accessibility as it avoids red/green issues.
parana_div_vio_grn = LinearSegmentedColormap.from_list(
    'parana_div_vio_grn',
    [parana_colors[4], '#f8f8f8', parana_colors[0]]
)

# --- NEW DIVERGING OPTION 2: Magenta -> Gray -> Green ---
# This provides a vibrant, high-contrast alternative.
parana_div_mag_grn = LinearSegmentedColormap.from_list(
    'parana_div_mag_grn',
    [parana_colors[3], '#f8f8f8', parana_colors[0]]
)

# --- NEW SEQUENTIAL OPTION 1: Cool Blue ---
# A rich, cool-toned map from a light blue, through violet, to your royal blue.
parana_seq_blue = LinearSegmentedColormap.from_list(
    'parana_seq_blue',
    ['#e0e8f5', parana_colors[4], parana_colors[2]]
)

# --- NEW SEQUENTIAL OPTION 2: Vibrant Plasma ---
# A multi-hue map going from yellow through magenta to violet.
parana_seq_plasma = LinearSegmentedColormap.from_list(
    'parana_seq_plasma',
    ['#fff5cc', parana_colors[5], parana_colors[3], parana_colors[4]]
)

# --- Register the new colormaps with Matplotlib ---

parana_viridis = LinearSegmentedColormap.from_list(
    'parana_viridis',
    [parana_colors[2], parana_colors[4], parana_colors[0], parana_colors[5]]
)

# --- NEW OPTION 2: Jet-like Colormap (Rainbow) ---
# This map creates a full-spectrum rainbow effect similar to 'jet':
# Azul Royal -> Verde Bandeira -> Ouro Ipê -> Vermelho Terra
parana_jet = LinearSegmentedColormap.from_list(
    'parana_jet',
    [parana_colors[2], parana_colors[0], parana_colors[5], parana_colors[1]]
)



# --- Option 2: Green -> Gray -> Yellow (As requested, with caution) ---
parana_div_grn_yel = LinearSegmentedColormap.from_list(
    'parana_div_grn_yel',
    [parana_colors[0], '#f8f8f8', parana_colors[5]]
)

# --- Option 3: Yellow -> Gray -> Blue (New Suggestion - High Contrast) ---
parana_div_yel_blu = LinearSegmentedColormap.from_list(
    'parana_div_yel_blu',
    [parana_colors[5], '#f8f8f8', parana_colors[2]]
)

# --- Option 4: Yellow -> Gray -> Violet (New Suggestion - High Contrast) ---
parana_div_yel_vio = LinearSegmentedColormap.from_list(
    'parana_div_yel_vio',
    [parana_colors[5], '#f8f8f8', parana_colors[4]]
)

# --- Register all the new colormaps ---

plt.colormaps.register(parana_div_grn_yel)
plt.colormaps.register(parana_div_yel_blu)
plt.colormaps.register(parana_div_yel_vio)

# A friendly note on 'jet': While vibrant, rainbow colormaps can sometimes distort
# data perception because the color transitions aren't uniform. For quantitative
# analysis, a viridis-style map is often preferred!

# --- Register the new colormaps with Matplotlib ---
plt.colormaps.register(parana_viridis)
plt.colormaps.register(parana_jet)

# Qualitative (for categorical data)
parana_qual = ListedColormap(parana_colors, name='parana_qual')


# 3. Register colormaps so Matplotlib can find them by name
plt.colormaps.register(parana_seq)
plt.colormaps.register(parana_seq_blue)
plt.colormaps.register(parana_seq_plasma)
plt.colormaps.register(parana_div)
plt.colormaps.register(parana_seq_rich)
plt.colormaps.register(parana_qual)
plt.colormaps.register(parana_div_vio_grn)
plt.colormaps.register(parana_div_mag_grn)

# 4. Define theme background and text colors
BACKGROUND_COLOR = '#FCFCFC'
TEXT_COLOR = '#212529'

def get_escape_basin_cmap_norm():
    """
    Creates and returns a specific 3-color colormap and normalization
    for plotting escape basins.
    """
    # Define the colors for the three states: escape down (-1), no escape (0), escape up (1)
    colors = [parana_colors[1], BACKGROUND_COLOR, parana_colors[-1]]
    cmap = ListedColormap(colors)

    # Define the boundaries for the colors
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


# 5. Master function to apply all theme settings
def aplicar_tema():
    """Applies the complete 'Paraná Vibrante' theme to Matplotlib."""
    
    # Set the default color cycle for plot lines
    plt.rcParams['axes.prop_cycle'] = cycler(color=parana_colors)

    # Set backgrounds
    plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR
    plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR

    # Set text, labels, and ticks
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['axes.edgecolor'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR

    
plt.rcParams.update({
"text.usetex": True,
"font.family": "serif", # Use Times New Roman or similar
"font.size": 10, # Base font size
"axes.labelsize": 12, # Axis labels
"xtick.labelsize": 10, # X-ticks
"ytick.labelsize": 10, # Y-ticks
"figure.dpi": 300, # High resolution
"figure.autolayout": False # Disable auto-layout (use constrained_layout instead)
})
