import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from jitcdde import jitcdde_lyap, y, t
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")

############################################
# 1) Mackey-Glass Dataset Class
############################################
class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task. """
    def __init__(self,
                 tau,
                 constant_past,
                 nmg=10,
                 beta=0.2,
                 gamma=0.1,
                 dt=1.0,
                 splits=(1000., 200.),  # 1000 train, 200 test
                 start_offset=0.,
                 seed_id=42
    ):
        super().__init__()
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.traintime = splits[0]
        self.testtime  = splits[1]
        self.start_offset = start_offset
        self.seed_id = seed_id
        self.maxtime = self.traintime + self.testtime + self.dt

        self.traintime_pts = round(self.traintime / self.dt)
        self.testtime_pts  = round(self.testtime / self.dt)
        self.maxtime_pts   = self.traintime_pts + self.testtime_pts + 1

        self.mackeyglass_specification = [
            self.beta * y(0, t - self.tau) / (1 + y(0, t - self.tau)**self.nmg) - self.gamma*y(0)
        ]
        self.generate_data()
        self.split_data()

    def generate_data(self):
        np.random.seed(self.seed_id)
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        self.mackeyglass_soln = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps        = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps_weights= torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)

        count = 0
        for time in torch.arange(self.DDE.t + self.start_offset,
                                 self.DDE.t + self.start_offset + self.maxtime,
                                 self.dt,
                                 dtype=torch.float64):
            value, lyap_val, weight_val = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count, 0] = value[0]
            lyaps[count, 0]         = lyap_val[0]
            lyaps_weights[count, 0] = weight_val
            count += 1

        self.total_var = torch.var(self.mackeyglass_soln[:, 0], unbiased=True)
        self.lyap_exp  = (lyaps.t() @ lyaps_weights / lyaps_weights.sum()).item()

    def split_data(self):
        self.ind_train = torch.arange(0, self.traintime_pts).numpy()
        self.ind_test  = torch.arange(self.traintime_pts, self.maxtime_pts-1).numpy()

    def __len__(self):
        return len(self.mackeyglass_soln) - 1

    def __getitem__(self, idx):
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)  # (1,1)
        target = self.mackeyglass_soln[idx, :]                          # (1,)
        return sample, target

#############################################
# 2) Create Time Series Dataset
#############################################
def create_time_series_dataset(
    data,
    train_indices,
    test_indices,
    lookback_window,
    forecasting_horizon,
    num_bins,
    MSE=False
):
    """
    Create training and testing datasets based on predefined indices.

    Parameters:
    - data: List of (sample, target) tuples.
    - train_indices: List or numpy array of training indices.
    - test_indices: List or numpy array of testing indices.
    - lookback_window: Number of past points to use.
    - forecasting_horizon: Number of steps ahead to predict.
    - num_bins: Number of bins for classification (ignored if MSE=True).
    - MSE: If True, perform regression; else, classification.

    Returns:
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for testing data.
    """
    # Extract scalar values from data_list
    x = np.array([point[0] for point in data])  # shape: (N,)
    y = np.array([point[1] for point in data])  # shape: (N,)

    x_processed_train = []
    y_processed_train = []

    x_processed_test = []
    y_processed_test = []

    # Function to check if a sequence is entirely within the desired indices
    def is_valid_sequence(start_idx, end_idx, valid_indices):
        return all(idx in valid_indices for idx in range(start_idx, end_idx))

    # Build sequences for training
    for i in train_indices:
        # Calculate the required range for the sequence
        start = i - lookback_window - forecasting_horizon + 1
        end = i
        if start < 0:
            continue  # Skip if the window goes beyond the data start
        # Ensure the target index does not exceed y's bounds
        target_idx = i + forecasting_horizon - 1
        if target_idx >= len(y):
            continue
        # Check if all indices in the window are within training indices
        if is_valid_sequence(start, end, train_indices):
            x_window = x[start : start + lookback_window]  # shape: (lookback_window,)
            y_value  = y[target_idx]                       # scalar
            x_processed_train.append(x_window)
            y_processed_train.append(y_value)

    # Convert to numpy arrays
    X_train = np.array(x_processed_train)  # shape: (num_train_samples, lookback_window)
    y_train = np.array(y_processed_train)  # shape: (num_train_samples,)

    # If classification, bin the labels
    if not MSE:
        bin_edges = np.linspace(np.min(y_train), np.max(y_train), num_bins - 1)
        y_train = np.digitize(y_train, bin_edges)

    # Create training data list with proper reshaping
    train_data = []
    for i in range(len(X_train)):
        x_squeezed = torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(-1)  # shape: (lookback_window, 1)
        y_val = torch.tensor(y_train[i], dtype=torch.float32)                     # scalar
        train_data.append((x_squeezed, y_val))

    # Repeat the process for testing data
    for i in test_indices:
        start = i - lookback_window - forecasting_horizon + 1
        end = i
        if start < 0:
            continue
        target_idx = i + forecasting_horizon - 1
        if target_idx >= len(y):
            continue
        if is_valid_sequence(start, end, test_indices):
            x_window = x[start : start + lookback_window]  # shape: (lookback_window,)
            y_value = y[target_idx]                       # scalar
            x_processed_test.append(x_window)
            y_processed_test.append(y_value)

    X_test = np.array(x_processed_test)  # shape: (num_test_samples, lookback_window)
    y_test = np.array(y_processed_test)  # shape: (num_test_samples,)

    if not MSE:
        bin_edges = np.linspace(np.min(y_test), np.max(y_test), num_bins - 1)
        y_test = np.digitize(y_test, bin_edges)

    test_data = []
    for i in range(len(X_test)):
        x_squeezed = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(-1)  # shape: (lookback_window, 1)
        y_val = torch.tensor(y_test[i], dtype=torch.float32)                     # scalar
        test_data.append((x_squeezed, y_val))

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader

###############################################
# 3) Define a 3-Layer RNN Model for Regression
###############################################
class RNN(nn.Module):
    """
    An RNN with three RNN layers and a linear output for time-series forecasting.
    """
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: (batch, seq_length, hidden_size)

        # Decode the last time step
        out = out[:, -1, :]        # (batch, hidden_size)
        out = self.fc(out)         # (batch, output_size)
        return out

###########################################
# 4) Training and Evaluation Functions
###########################################
def train_rnn(model, train_loader, num_epochs=50, lr=1e-3, device='cpu'):
    """
    Train the RNN model.

    Parameters:
    - model: The RNN model to train.
    - train_loader: DataLoader for training data.
    - num_epochs: Number of epochs to train.
    - lr: Learning rate.
    - device: Device to train on ('cpu' or 'cuda').
    """
    criterion = torch.nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float().to(device)  # (batch_size, lookback_window, 1)
            y_batch = y_batch.float().to(device).unsqueeze(1)  # (batch_size, 1)

            optimizer.zero_grad()
            mean = model(x_batch)
            # Fixed variance (tensor of ones)
            variance = torch.ones_like(mean).to(device)
            loss = criterion(mean, y_batch, variance)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}")

def evaluate_rnn(model, test_loader, device='cpu', verbose=True):
    """
    Evaluate the RNN model.

    Parameters:
    - model: The trained RNN model.
    - test_loader: DataLoader for testing data.
    - device: Device to evaluate on ('cpu' or 'cuda').
    - verbose: If True, print the MSE loss.

    Returns:
    - avg_loss: Average Gaussian NLLL loss on the test set.
    """
    model.eval()
    criterion = torch.nn.GaussianNLLLoss()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device).unsqueeze(1)

            mean = model(x_batch)
            # Fixed variance (tensor of ones)
            variance = torch.ones_like(mean).to(device)
            loss = criterion(mean, y_batch, variance)
            total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    if verbose:
        print(f"Test Gaussian NLL Loss: {avg_loss:.6f}")
    return avg_loss

def forecast(model, init_sequence, steps=5, device='cpu'):
    """
    Forecast steps future values given an initial sequence (lookback_window x 1).

    Parameters:
    - model: The trained RNN model.
    - init_sequence: Tensor containing the initial sequence.
    - steps: Number of future steps to forecast.
    - device: Device to perform forecasting on ('cpu' or 'cuda').

    Returns:
    - forecasts: List of forecasted values.
    """
    model.eval()
    forecasts = []

    current_seq = init_sequence.clone().float().to(device).unsqueeze(0)  # shape: (1, lookback_window, 1)
    with torch.no_grad():
        for _ in range(steps):
            out = model(current_seq)           # shape: (1,1)
            next_val = out[:, 0].item()        # scalar
            forecasts.append(next_val)

            # Prepare the next input sequence
            next_input = torch.tensor([[next_val]], dtype=torch.float32).to(device)
            current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(0)], dim=1)

    return forecasts

#####################################
# 5) Experimentation and Plotting
#####################################
def run_experiments():
    # Configuration
    tau = 17
    constant_past = 1.2
    splits = (1000., 200.)  # 1000 train, 200 test
    seed_id = 42

    lookback_windows = [3, 5, 7]
    forecasting_horizons = list(range(1, 26))  # 1 to 25
    num_bins = 5
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

    # Initialize a dictionary to store MSE losses
    mse_results = {lb: [] for lb in lookback_windows}

    # Iterate over each lookback window
    for lb in lookback_windows:
        print(f"\n=== Lookback Window: {lb} ===")
        mse_horizon = []

        # Iterate over each forecasting horizon
        for fh in forecasting_horizons:
            print(f"  Forecasting Horizon: {fh}", end='\r')

            # Create DataLoaders using predefined train and test indices
            train_loader, test_loader = create_time_series_dataset(
                data_list,
                train_indices=dataset.ind_train,
                test_indices=dataset.ind_test,
                lookback_window=lb,
                forecasting_horizon=fh,
                num_bins=num_bins,
                MSE=True
            )

            # Initialize the model
            model = RNN(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=3)
            model.to(device)

            # Train the model
            train_rnn(model, train_loader, num_epochs=num_epochs, lr=lr, device=device)

            # Evaluate the model on the test set
            test_loss = evaluate_rnn(model, test_loader, device=device, verbose=False)
            mse_horizon.append(test_loss)

        mse_results[lb] = mse_horizon

        # Plotting for the current lookback window
        plt.figure(figsize=(10, 6))
        plt.plot(forecasting_horizons, mse_results[lb], marker='o', label=f"Lookback={lb}")
        plt.title(f"Gaussian NLLL vs Forecasting Horizon (Lookback={lb})")
        plt.xlabel("Forecasting Horizon")
        plt.ylabel("Gaussian NLLL Loss")
        plt.xticks(forecasting_horizons)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Optional: Print a summary table of results
    print("\n=== Summary of Gaussian NLLL Losses ===")
    for lb in lookback_windows:
        print(f"\nLookback Window: {lb}")
        for fh, mse in zip(forecasting_horizons, mse_results[lb]):
            if not np.isnan(mse):
                print(f"  Forecasting Horizon {fh}: Gaussian NLLL Loss = {mse:.6f}")
            else:
                print(f"  Forecasting Horizon {fh}: Skipped due to insufficient test samples.")

if __name__ == "__main__":
    run_experiments()