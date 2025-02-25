import matplotlib.pyplot as plt

def plot_forecasting_loss(mcd_data_dict, baseline_data_dict):
    horizons = list(map(int, mcd_data_dict.keys()))
    mcd_mse_values = [mcd_data_dict[str(h)][0] for h in horizons]  # Extract MSE values for MCD
    mcd_mae_values = [mcd_data_dict[str(h)][1] for h in horizons]  # Extract MAE values for MCD
    
    baseline_mse_values = [baseline_data_dict[str(h)][0] for h in horizons]  # Extract MSE values for baseline
    baseline_mae_values = [baseline_data_dict[str(h)][1] for h in horizons]  # Extract MAE values for baseline

    plt.figure(figsize=(10, 5))
    plt.plot(horizons, mcd_mse_values, marker='o', linestyle='-', label='MCD MSE Loss')
    plt.plot(horizons, mcd_mae_values, marker='s', linestyle='-', label='MCD MAE Loss')
    plt.plot(horizons, baseline_mse_values, marker='o', linestyle='--', label='Baseline MSE Loss')
    plt.plot(horizons, baseline_mae_values, marker='s', linestyle='--', label='Baseline MAE Loss')
    
    plt.xlabel("Forecasting Horizon")
    plt.ylabel("Loss")
    plt.title("Forecasting Loss vs. Horizon (MCD vs. Baseline)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_forecasting_losses(all_results, baseline_data_dict, horizons):
    plt.figure(figsize=(12, 6))
    
    # Plot baseline in black
    baseline_mse_values = [baseline_data_dict[str(h)][0] for h in horizons]
    baseline_mae_values = [baseline_data_dict[str(h)][1] for h in horizons]
    plt.plot(horizons, baseline_mse_values, 'k--', label='Baseline MSE')
    plt.plot(horizons, baseline_mae_values, 'k-', label='Baseline MAE')
    
    # Plot each hyperparameter combination
    for run_id, (config, mcd_data_dict) in all_results.items():
        mse_values = [mcd_data_dict[str(h)][0] for h in horizons]
        mae_values = [mcd_data_dict[str(h)][1] for h in horizons]
        config_label = f'ID {run_id}: {config["hs"]}, {config["lr"]}, {config["num_epochs"]}, {config["optimizer"]}, {config["lookback_window"]}'
        plt.plot(horizons, mse_values, '--', label=f'{config_label} MSE')
        plt.plot(horizons, mae_values, '-', label=f'{config_label} MAE')
    
    plt.xlabel("Forecasting Horizon")
    plt.ylabel("Loss")
    plt.title("Forecasting Loss vs. Horizon (MCD vs. Baseline)")
    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


