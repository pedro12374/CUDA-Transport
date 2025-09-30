import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import parana_theme as tema

tema.aplicar_tema()

# ==============================================================================
# == CORE PLOTTING FUNCTIONS
# ==============================================================================

def _plot_escape_time(ax, data, title, bounds):
    """Internal function to plot a 2D escape time map."""
    data[data == -1] = np.nan
    im = ax.imshow(data, origin='lower',
                   norm=LogNorm(), cmap=tema.parana_jet,
                   extent=bounds) # Use the provided bounds
    ax.set_title(title)
    return im, "Escape Time (t)"

def _plot_lyapunov(ax, data, title, bounds):
    """Internal function to plot a 2D Lyapunov exponent map."""
    im = ax.imshow(data, origin='lower', cmap=tema.parana_seq_plasma,
                   extent=bounds) # Use the provided bounds
    ax.set_title(title)
    return im, r"Max Lyapunov Exponent ($\lambda$)"

def _plot_stroboscopic(ax, data, title, bounds):
    """Internal function to plot a stroboscopic map."""
    all_x = data[:, :, 0].flatten()
    all_y = data[:, :, 1].flatten()
    ax.plot(all_x, all_y, ',', color='black', markersize=0.2, alpha=0.6, rasterized=True)
    ax.set_title(title)
    ax.set_xlim(bounds[0], bounds[1]) # Use bounds for x-axis
    ax.set_ylim(bounds[2], bounds[3]) # Use bounds for y-axis
    ax.set_aspect('equal', adjustable='box')
    return None, None

# ==============================================================================
# == PUBLIC MASTER FUNCTION
# ==============================================================================

def generate_mosaic_plot(h5_file, output_pdf, plot_type, dset_prefix, params_list, bounds=None):
    """
    Generates and saves a mosaic plot based on the provided configuration.
    'bounds' should be a list/tuple: [xmin, xmax, ymin, ymax]
    """
    # --- Default Bounds ---
    if bounds is None:
        bounds = [-np.pi, np.pi, -np.pi, np.pi] # Default if not specified

    # --- Setup Figure ---
    num_plots = len(params_list)
    # ... (rest of the figure setup is the same)
    if num_plots <= 2:
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5), constrained_layout=True)
        if num_plots == 1: axs = [axs]
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        axs = axs.flatten()

    print(f"Generating '{plot_type}' plot from '{h5_file}'...")

    # --- Loop and Plot ---
    im = None
    for i, p_val in enumerate(params_list):
        if i >= len(axs): break
        ax = axs[i]
        dset_name = f"{dset_prefix}_{p_val:.2f}" if isinstance(p_val, float) else f"{dset_prefix}_{p_val:.6f}"
        
        try:
            with h5py.File(h5_file, 'r') as f:
                data = f[dset_name][:]

            title = f"{dset_prefix.split('_')[-1]} = {p_val}"
            
            # Pass the bounds to the internal plotting function
            if plot_type == 'escape':
                im, cbar_label = _plot_escape_time(ax, data, title, bounds)
            elif plot_type == 'lyapunov':
                im, cbar_label = _plot_lyapunov(ax, data, title, bounds)
            elif plot_type == 'strobo':
                im, cbar_label = _plot_stroboscopic(ax, data, title, bounds)

        except KeyError:
            print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
            continue

        # ... (rest of the labeling logic is the same)
        if i % 2 == 0: ax.set_ylabel('Y')
        if i // 2 == 1 or num_plots <= 2: ax.set_xlabel('X')


    # --- Final Touches ---
    if im is not None:
        fig.colorbar(im, ax=axs.tolist(), location='right', shrink=0.6, label=cbar_label)

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Plot successfully saved to: {output_pdf}")
    plt.close(fig)

def generate_single_plot(h5_file, output_pdf, plot_type, dset_name, title=None, bounds=None):
    """
    Generates and saves a single plot for a specific dataset.
    """
    # --- Default Bounds ---
    if bounds is None:
        bounds = [-np.pi, np.pi, -np.pi, np.pi]

    # --- Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    print(f"Generating single '{plot_type}' plot from '{dset_name}'...")

    im = None # Initialize im
    try:
        with h5py.File(h5_file, 'r') as f:
            data = f[dset_name][:]

        plot_title = title if title else dset_name
        
        # Call the appropriate internal plotting function
        if plot_type == 'escape':
            im, cbar_label = _plot_escape_time(ax, data, plot_title, bounds)
        elif plot_type == 'lyapunov':
            im, cbar_label = _plot_lyapunov(ax, data, plot_title, bounds)
        elif plot_type == 'strobo':
            im, cbar_label = _plot_stroboscopic(ax, data, plot_title, bounds)

    except KeyError:
        print(f"Error: Dataset '{dset_name}' not found in '{h5_file}'.")
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- Final Touches ---
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    if im is not None:
        fig.colorbar(im, ax=ax, location='right', shrink=0.8, label=cbar_label)

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Plot successfully saved to: {output_pdf}")
    plt.close(fig)

def generate_individual_plots(h5_file, output_dir, plot_type, dset_prefix, params_list, title_prefix="", bounds=None):
    """
    Generates and saves a separate plot for each parameter in a list.
    """
    print(f"\n--- Generating Individual '{plot_type}' Plots ---")
    
    for p_val in params_list:
        # 1. Construct the specific dataset name for this parameter
        dset_name = f"{dset_prefix}_{p_val:.2f}" if isinstance(p_val, float) else f"{dset_prefix}_{p_val:.6f}"
        
        # 2. Create a unique, descriptive output filename for this plot
        output_pdf = f"{output_dir}/{plot_type}_{dset_name}.pdf"
        
        # 3. Create a descriptive title for the plot
        title = f"{title_prefix} ({dset_prefix}={p_val})" if title_prefix else dset_name

        # 4. Call the existing single_plot function with the generated info
        generate_single_plot(
            h5_file=h5_file,
            output_pdf=output_pdf,
            plot_type=plot_type,
            dset_name=dset_name,
            title=title,
            bounds=bounds
        )