import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # 1. Import the curve_fit function

# --- Configuration ---
K_VALUE_TO_ANALYZE = 0.971635
FIT_START_ITERATION = 10
FIT_END_ITERATION = 8000

# 2. Define the model function we want to fit to the data.
#    't' is the independent variable (iterations).
#    'D' and 'alpha' are the parameters we want to find.
def power_law(t, D, alpha):
    """Power-law model for MSD: MSD = D * t^alpha"""
    return D * (t**alpha)

# --- Analysis ---
dset_name = f"K_{K_VALUE_TO_ANALYZE:.6f}"
print(f"Analyzing diffusion for dataset: '{dset_name}' using scipy.optimize.curve_fit\n")

try:
    with h5py.File('../dat/msd_p.h5', 'r') as f:
        if dset_name not in f:
            raise KeyError(f"Dataset '{dset_name}' not found in msd_p.h5.")
        
        msd_data = f[dset_name][:]
        iterations = np.arange(len(msd_data))

    # Slice the data to the fitting range (ignoring t=0)
    fit_slice = slice(FIT_START_ITERATION, FIT_END_ITERATION)
    t_fit = iterations[fit_slice]
    msd_fit = msd_data[fit_slice]

    # 3. Perform the curve fit.
    #    popt: array of the optimal parameters found [D, alpha]
    #    pcov: the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
    popt, pcov = curve_fit(power_law, t_fit, msd_fit)
    
    # Extract the optimal parameters
    D = popt[0]
    alpha = popt[1]
    
    # Calculate the standard errors for the parameters
    errors = np.sqrt(np.diag(pcov))
    D_error = errors[0]
    alpha_error = errors[1]

    print("--- Fit Results ---")
    print(f"Diffusion Exponent (α): {alpha:.4f} ± {alpha_error:.4f}")
    print(f"Diffusion Coefficient (D): {D:.4e} ± {D_error:.4e}")
    print("---------------------\n")

    # 4. Generate the best-fit line using the fitted parameters
    fit_line = power_law(t_fit, D, alpha)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.loglog(iterations[1:], msd_data[1:], 'o', markersize=3, label='Raw MSD Data')
    plt.loglog(t_fit, fit_line, 'r-', linewidth=2, label=f'Fit: α = {alpha:.3f} ± {alpha_error:.3f}')
    
    plt.axvspan(FIT_START_ITERATION, FIT_END_ITERATION, color='gray', alpha=0.2, label='Fit Region')

    plt.title(f'MSD Analysis for K = {K_VALUE_TO_ANALYZE}')
    plt.xlabel('Iteration Number (t)')
    plt.ylabel('MSD(p)')
    plt.legend()
    
    output_filename = f"diffusion_analysis_scipy_K_{K_VALUE_TO_ANALYZE}.png"
    plt.savefig(output_filename)
    print(f"Saved analysis plot to {output_filename}")
    
    plt.show()

except (FileNotFoundError, KeyError, IndexError) as e:
    print(f"An error occurred: {e}")