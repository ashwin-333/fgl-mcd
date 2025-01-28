import torch
from utils import MackeyGlass, create_time_series_dataset, RNN, train_rnn, evaluate_rnn, calibrate_uncertainty
from visualize import plot_results, plot_uncertainty


def run_experiments():
    """
    Run the main experiment pipeline.
    """
    # Configuration
    tau = 17
    constant_past = 1.2
    splits = (1000., 200.)  # 1000 train, 200 test
    seed_id = 42

    lookback_windows = [3, 5, 7]
    forecasting_horizons = list(range(1, 26))  # 1 to 25
    num_epochs = 100
    lr = 1e-3
    hidden_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate Mackey-Glass dataset with updated splits
    dataset = MackeyGlass(
        tau=tau,
        constant_past=constant_past,
        splits=splits,
        seed_id=seed_id
    )

    # Convert to list of (sample, target) pairs (as scalars)
    data_list = []
    for i in range(len(dataset)):
        sample, target = dataset[i]
        sample_val = sample.numpy()[0, 0]  # Extract scalar value
        target_val = target.numpy()[0]     # Extract scalar value
        data_list.append((sample_val, target_val))

    # Initialize a dictionary to store Gaussian NLLL losses
    results = {lb: [] for lb in lookback_windows}

    # Iterate over each lookback window
    for lb in lookback_windows:
        print(f"\n=== Lookback Window: {lb} ===")
        horizon_losses = []

        # Iterate over each forecasting horizon
        for fh in forecasting_horizons:
            print(f"  Forecasting Horizon: {fh}", end='\r')

            # Create DataLoaders using predefined train and test indices
            train_loader, test_loader = create_time_series_dataset(
                data_list,
                dataset.ind_train,
                dataset.ind_test,
                lookback_window=lb,
                forecasting_horizon=fh,
                num_bins=5,
                MSE=True
            )

            # Initialize the model
            model = RNN(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=3)
            model.to(device)

            # Train the model
            train_rnn(model, train_loader, num_epochs=num_epochs, lr=lr, device=device)

            # Evaluate the model on the test set
            test_loss, preds, true_values = evaluate_rnn(model, test_loader, device=device, verbose=False)

            #Calibrate and plot uncertainty
            calibrated_preds = calibrate_uncertainty(preds, true_values)
            plot_uncertainty(preds, calibrated_preds, true_values)

            horizon_losses.append(test_loss)

        results[lb] = horizon_losses

    # Plotting results
    plot_results(results, lookback_windows, forecasting_horizons)

    # Optional: Print a summary table of results
    print("\n=== Summary of Gaussian NLLL Losses ===")
    for lb in lookback_windows:
        print(f"\nLookback Window: {lb}")
        for fh, loss in zip(forecasting_horizons, results[lb]):
            if not torch.isnan(torch.tensor(loss)):
                print(f"  Forecasting Horizon {fh}: Gaussian NLLL Loss = {loss:.6f}")
            else:
                print(f"  Forecasting Horizon {fh}: Skipped due to insufficient test samples.")

if __name__ == "__main__":
    run_experiments()
