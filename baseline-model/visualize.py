import matplotlib.pyplot as plt


def plot_results(results, lookback_windows, forecasting_horizons):
    """
    Plot the Gaussian NLL Loss vs Forecasting Horizon for different lookback windows.
    """
    for lb in lookback_windows:
        plt.figure(figsize=(10, 6))
        plt.plot(forecasting_horizons, results[lb], marker='o', label=f"Lookback={lb}")
        plt.title(f"Gaussian NLLL vs Forecasting Horizon (Lookback={lb})")
        plt.xlabel("Forecasting Horizon")
        plt.ylabel("Gaussian NLLL Loss")
        plt.xticks(forecasting_horizons)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
