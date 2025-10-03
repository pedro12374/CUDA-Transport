import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, BoundaryNorm, ListedColormap
from scipy.optimize import curve_fit

import parana_theme as tema

tema.aplicar_tema()

# ==============================================================================
# == CORE PLOTTING FUNCTIONS
# ==============================================================================

def _plot_escape_time(ax, data, title, bounds):
    """Internal function to plot a 2D escape time map."""
    data[data == -1] = np.nan
    if np.all(np.isnan(data)) or len(np.unique(data[~np.isnan(data)])) <= 1:
        # If there's no range, use a simple linear norm instead of log
        norm = None 
        print(f"    - Warning: No variation in escape time data for '{title}'. Using linear scale.")
    else:
        # Otherwise, use the logarithmic norm as intended
        norm = LogNorm()
    im = ax.imshow(data.T, origin='lower',
                   norm=norm, cmap=tema.parana_jet,
                   extent=bounds,aspect='auto') # Use the provided bounds
    ax.set_title(title)
    return im, "Escape Time (t)"

def _plot_escape_basin(ax, data, title, bounds, cmap_config=None):
    """Internal function to plot escape basins with customizable colormap."""
    
    if cmap_config:
        # Use user-provided configuration
        colors = cmap_config['colors']
        norm_bounds = cmap_config['bounds']
        cbar_ticks = cmap_config.get('ticks') # .get() makes it optional
    else:
        # Fallback to the default theme for basins -1, 0, 1
        colors = [tema.VERMELHO, tema.BACKGROUND_COLOR, tema.AMARELO]
        norm_bounds = [-1.5, -0.5, 0.5, 1.5]
        cbar_ticks = [-1, 0, 1]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(norm_bounds, cmap.N)

    im = ax.imshow(data.T, origin='lower', cmap=cmap, norm=norm, extent=bounds,aspect='auto')
    ax.set_title(title)
    
    return im, {"label": "Escape Basin", "ticks": cbar_ticks}

def _plot_lyapunov(ax, data, title, bounds):
    """Internal function to plot a 2D Lyapunov exponent map."""
    im = ax.imshow(data.T, origin='lower', cmap=tema.parana_seq_plasma,
                   extent=bounds,aspect='auto') # Use the provided bounds
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
def _plot_msd(ax, data, title, bounds=None): # bounds is unused but keeps signature consistent
    """Internal function to plot MSD data on a log-log scale."""
    t = np.arange(len(data)) * 0.01 # Assuming DT=0.01 from main.cu
    # Power-law model for fitting: MSD = D * t^alpha
    def power_law(t, D, alpha):
        return D * (t**alpha)

    # Fit the data (ignoring the first few points)
    fit_start = 1
    if len(t) > fit_start:
        popt, _ = curve_fit(power_law, t[fit_start:], data[fit_start:])
        alpha = popt[1]
        fit_label = f'$\\alpha \\approx {alpha:.2f}$'
        ax.plot(t[fit_start:], power_law(t[fit_start:], *popt), 'r--', label=fit_label)
    
    ax.loglog(t, data)
    ax.set_title(title)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$\langle \Delta y^2(t) \rangle$")
    ax.legend()
    return None, None # No shared colorbar
def _plot_displacement(ax, data, title, bounds):
    """Internal function to plot a 2D displacement map."""
    # We'll plot the displacement in the Y-dimension (index 1)
    disp_y = data[:, :, 1]
    
    # Use a diverging colormap centered at zero
    limit = np.max(np.abs(disp_y))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    
    im = ax.imshow(disp_y.T, origin='lower', cmap=tema.parana_div_yel_blu,
                   norm=norm, extent=bounds,aspect='auto')
    ax.set_title(title)
    return im, r"Final Displacement ($\Delta y$)"
def _plot_total_displacement(ax, data, title, bounds):
    """Internal function to plot a 2D total displacement map."""
    # Use a sequential colormap since magnitude is always positive
    im = ax.imshow(data.T, origin='lower', cmap=tema.parana_seq_plasma,
                   norm=LogNorm(), # Log scale is often good for displacement magnitudes
                   extent=bounds,aspect='auto')
    ax.set_title(title)
    return im, r"Total Displacement Magnitude $\sqrt{\Delta x^2 + \Delta y^2}$"

# ==============================================================================
# == PUBLIC MASTER FUNCTION
# ==============================================================================

def generate_mosaic_plot(h5_file, output_pdf, plot_type, dset_prefix, params_list, bounds=None, basin_cmap_config=None):

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
            elif plot_type == 'displacement':
                im, cbar_label = _plot_displacement(ax, data, title, bounds)
            elif plot_type == 'total_displacement': 
                im, cbar_label = _plot_total_displacement(ax, data, title, bounds)
            elif plot_type == 'msd':
                im, cbar_label = _plot_msd(ax, data, title, bounds)
            elif plot_type == 'basin':
                # Pass the new config to the internal function
                im, cbar_label = _plot_escape_basin(ax, data, title, bounds, basin_cmap_config)
            else:
                print(f"Error: Unknown plot type '{plot_type}'.")
                return


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
    
def generate_parameter_matrix_plot(h5_file, output_pdf, plot_type, dset_prefix, row_params, col_params, row_prefix, col_prefix, bounds=None, basin_cmap_config=None):
    """
    Generates a 2D matrix of plots for a two-parameter sweep.
    """
    # --- Default Bounds ---
    if bounds is None:
        bounds = [-np.pi, np.pi, -np.pi, np.pi]

    # --- Setup Figure ---
    num_rows = len(row_params)
    num_cols = len(col_params)
    fig, axs = plt.subplots(num_rows, num_cols, 
                            figsize=(3 * num_cols, 3 * num_rows), 
                            constrained_layout=True,
                            sharex=True, sharey=True)

    print(f"Generating '{plot_type}' parameter matrix plot from '{h5_file}'...")

    im = None
    for i, r_val in enumerate(row_params):
        for j, c_val in enumerate(col_params):
            ax = axs[i, j]
            
            # Construct the dataset name based on the C++ format
            dset_name = f"{dset_prefix}_{col_prefix}_{c_val:.2f}_{row_prefix}_{r_val:.2f}"
            
            try:
                with h5py.File(h5_file, 'r') as f:
                    data = f[dset_name][:]

                # Call the appropriate internal plotting function
                if plot_type == 'escape':
                    im, cbar_label = _plot_escape_time(ax, data, title, bounds)
                elif plot_type == 'lyapunov':
                    im, cbar_label = _plot_lyapunov(ax, data, title, bounds)
                elif plot_type == 'strobo':
                    im, cbar_label = _plot_stroboscopic(ax, data, title, bounds)
                elif plot_type == 'displacement':
                    im, cbar_label = _plot_displacement(ax, data, title, bounds)
                elif plot_type == 'total_displacement': 
                    im, cbar_label = _plot_total_displacement(ax, data, title, bounds)
                elif plot_type == 'msd':
                    im, cbar_label = _plot_msd(ax, data, title, bounds)
                elif plot_type == 'basin':
             # Pass the new config to the internal function
                    im, cbar_label = _plot_escape_basin(ax, data, plot_title, bounds, basin_cmap_config)
                # Add other plot types here if needed
                else:
                    print(f"Error: Unknown plot type '{plot_type}'.")
                    return

            except KeyError:
                print(f"Warning: Dataset '{dset_name}' not found. Skipping.")
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
                continue

            # --- Labeling ---
            if i == 0: # Top row
                ax.set_title(f"${col_prefix} = {c_val}$")
            if j == 0: # Left-most column
                ax.set_ylabel(f"${row_prefix} = {r_val}$")

    # --- Final Touches ---
    if im is not None:
        fig.colorbar(im, ax=axs, location='right', aspect=40, shrink=0.8, label=cbar_label)

    fig.supxlabel('X')
    fig.supylabel('Y')
    
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Parameter matrix plot successfully saved to: {output_pdf}")
    plt.close(fig)

def generate_single_plot(h5_file, output_pdf, plot_type, dset_name, title=None, bounds=None, basin_cmap_config=None):

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
            im, cbar_label = _plot_escape_time(ax, data, title, bounds)
        elif plot_type == 'lyapunov':
            im, cbar_label = _plot_lyapunov(ax, data, title, bounds)
        elif plot_type == 'strobo':
            im, cbar_label = _plot_stroboscopic(ax, data, title, bounds)
        elif plot_type == 'displacement':
            im, cbar_label = _plot_displacement(ax, data, title, bounds)
        elif plot_type == 'total_displacement': 
            im, cbar_label = _plot_total_displacement(ax, data, title, bounds)
        elif plot_type == 'msd':
            im, cbar_label = _plot_msd(ax, data, title, bounds)
        elif plot_type == 'basin':
             # Pass the new config to the internal function
            im, cbar_label = _plot_escape_basin(ax, data, plot_title, bounds, basin_cmap_config)
        else:
            print(f"Error: Unknown plot type '{plot_type}'.")
            return

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

def generate_individual_plots(h5_file, output_dir, plot_type, dset_prefix, params_list, title_prefix="", bounds=None, basin_cmap_config=None):
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
            bounds=bounds,
            basin_cmap_config=basin_cmap_config # or pass a config if
        )
def generate_individual_matrix_plots(h5_file, output_dir, plot_type, dset_prefix, row_params, col_params, row_prefix, col_prefix, bounds=None, basin_cmap_config=None):
    """
    Generates a separate plot file for each combination in a 2D parameter sweep.
    """
    print(f"\n--- Generating Individual '{plot_type}' Plots for 2D Sweep ---")
    
    # Iterate through every combination of row and column parameters
    for r_val in row_params:
        for c_val in col_params:
            # 1. Construct the specific dataset name
            dset_name = f"{dset_prefix}_{col_prefix}_{c_val:.4f}_{row_prefix}_{r_val:.4f}"
            
            # 2. Create a unique, descriptive output filename
            output_pdf = f"{output_dir}/{plot_type}_{col_prefix}_{c_val:.4f}_{row_prefix}_{r_val:.4f}.pdf"
            
            # 3. Create a descriptive title
            title = f"{plot_type.capitalize()}: ${col_prefix}={c_val}$, ${row_prefix}={r_val}$"

            # 4. Call the existing single_plot function
            generate_single_plot(
                h5_file=h5_file,
                output_pdf=output_pdf,
                plot_type=plot_type,
                dset_name=dset_name,
                title=title,
                bounds=bounds,
                basin_cmap_config=basin_cmap_config # or pass a config if
            )