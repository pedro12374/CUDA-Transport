import plotting_lib as pl
import numpy as np # Needed for np.pi

# ==============================================================================
# == DEFINE ALL PLOTS TO BE GENERATED
# ==============================================================================

plots_to_generate = [
    {
        "h5_file": "../dat/horton_escape_analysis_vs_A2.h5",
        "output_pdf": "../plots/horton_escape_A2_mosaic.pdf",
        "plot_type": "escape",
        "dset_prefix": "Escape_A2",
        "params_list": [0.1, 0.5, 1.0, 1.5],
        "bounds": [-np.pi, np.pi, -np.pi, np.pi] # Bounds for the Horton system
    },
    {
        "h5_file": "../dat/three_wave_stroboscopic_vs_A2.h5",
        "output_pdf": "../plots/horton_strobo_A2_mosaic.pdf",
        "plot_type": "strobo",
        "dset_prefix": "Strobo_A2",
        "params_list": [0.1, 0.5, 1.0, 1.5],
        "bounds": [-np.pi, np.pi, -np.pi, np.pi] # Bounds for the Horton system
    },
    {
        "h5_file": "../dat/lyapunov_exponents.h5",
        "output_pdf": "../plots/standard_map_lyapunov_mosaic.pdf",
        "plot_type": "lyapunov",
        "dset_prefix": "K",
        "params_list": [0.5, 0.971635, 1.5, 6.47],
        "bounds": [0, 2*np.pi, -np.pi, np.pi] # Different bounds for the Standard Map!
    }
]

# ==============================================================================
# == EXECUTION LOOP
# ==============================================================================

if __name__ == '__main__':
    for plot_config in plots_to_generate:
        pl.generate_mosaic_plot(
            h5_file=plot_config["h5_file"],
            output_pdf=plot_config["output_pdf"],
            plot_type=plot_config["plot_type"],
            dset_prefix=plot_config["dset_prefix"],
            params_list=plot_config["params_list"],
            bounds=plot_config.get("bounds") # Use .get() for safety if bounds are omitted
        )

    print("\nAll plots have been generated.")