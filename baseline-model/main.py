import os
import pickle
import torch
from utils import MackeyGlass, create_time_series_dataset, RNN, train_rnn, evaluate_rnn, calibrate_uncertainty
from visualize import plot_results, plot_uncertainty

MODEL_PATH_TEMPLATE = "teacher_model_lb{lb}.pth"
DATA_PATH = "mackey_glass_data.pkl"

def save_mackey_glass_data(tau, constant_past, splits, seed_id, filepath):
    """
    Generates Mackey-Glass data and saves it to a .pkl file.
    """
    print(f"Generating Mackey-Glass data...")
    dataset = MackeyGlass(
        tau=tau,
        constant_past=constant_past,
        splits=splits,
        seed_id=seed_id
    )
    data_list = [(sample.numpy()[0, 0], target.numpy()[0]) for sample, target in dataset]

    with open(filepath, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Saved Mackey-Glass data to {filepath}")
    return data_list

def load_mackey_glass_data(filepath):
    """
    Loads Mackey-Glass data from a .pkl file.
    """
    print(f"Loading Mackey-Glass data from {filepath}...")
    with open(filepath, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def run_experiments():
    """
    Train or load teacher models for different lookback windows, evaluate, and reuse them if already saved.
    """
    # Configuration
    tau = 17
    constant_past = 1.2
    splits = (800., 200.)  # 1000 train, 200 test
    seed_id = 42

    lookback_windows = [3, 5, 7]  # Three different configurations
    forecasting_horizons = [1]  # Next-step forecasting only
    num_epochs = 100
    lr = 1e-3
    hidden_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if Mackey-Glass data exists, otherwise generate and save it
    if not os.path.exists(DATA_PATH):
        data_list = save_mackey_glass_data(tau, constant_past, splits, seed_id, DATA_PATH)
    else:
        data_list = load_mackey_glass_data(DATA_PATH)

    results = {}

    for lb in lookback_windows:
        print(f"\n=== Lookback Window: {lb} ===")

        # Define the model path for the current lookback window
        model_path = MODEL_PATH_TEMPLATE.format(lb=lb)

        # Initialize a dictionary to store results for the current LB
        results[lb] = []

        for fh in forecasting_horizons:
            print(f"  Forecasting Horizon: {fh}", end='\r')

            # Create DataLoaders for the current LB and FH
            train_loader, test_loader = create_time_series_dataset(
                data_list,
                range(800),  # Using predefined splits
                range(800, 1000),
                lookback_window=lb,
                forecasting_horizon=fh,
                num_bins=5,
                MSE=True
            )

            # Initialize the teacher model
            model = RNN(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=3)
            model.to(device)

            # Check if the saved model exists for the current LB
            if os.path.exists(model_path):
                print(f"\nLoading the saved model for LB={lb} from {model_path}...")
                model.load_state_dict(torch.load(model_path))
                model.eval()  # Set model to evaluation mode
            else:
                # Train the teacher model
                print(f"\nTraining Teacher Model (LB={lb}, FH={fh})...")
                train_rnn(model, train_loader, num_epochs=num_epochs, lr=lr, device=device)

                # Save the trained model
                print(f"Saving the trained model for LB={lb} to {model_path}...")
                torch.save(model.state_dict(), model_path)

            # Evaluate the teacher model
            print("\nEvaluating the Teacher Model...")
            test_loss, preds, true_values = evaluate_rnn(model, test_loader, device=device, verbose=False)

            print(f"\n=== Teacher Model Test Loss (LB={lb}, FH={fh}): {test_loss:.6f} ===")

            # Calibrate and plot uncertainty
            calibrated_preds, ir_lower, ir_upper = calibrate_uncertainty(preds, true_values)
            plot_uncertainty(preds, calibrated_preds, true_values, ir_lower, ir_upper, fh, lb)

            # Store the test loss for the current FH
            results[lb].append(test_loss)

    # Plotting results
    plot_results(results, lookback_windows, forecasting_horizons)

    # Print Summary of Losses
    print("\n=== Summary of Gaussian NLL Losses ===")
    for lb in lookback_windows:
        print(f"\nLookback Window: {lb}")
        for fh, loss in zip(forecasting_horizons, results[lb]):
            if not torch.isnan(torch.tensor(loss)):
                print(f"  Forecasting Horizon {fh}: Gaussian NLL Loss = {loss:.6f}")
            else:
                print(f"  Forecasting Horizon {fh}: Skipped due to insufficient test samples.")

if __name__ == "__main__":
    run_experiments()
