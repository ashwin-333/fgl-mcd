import numpy as np
import torch
import math
from jitcdde import jitcdde_lyap, y, t
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

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
                 splits=(30., 10.),
                 start_offset=0.,
                 seed_id=0
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

        self.traintime_pts = round(self.traintime/self.dt)
        self.testtime_pts  = round(self.testtime/self.dt)
        self.maxtime_pts   = self.traintime_pts + self.testtime_pts + 1

        self.mackeyglass_specification = [
            self.beta * y(0, t - self.tau) / (1 + y(0, t - self.tau)**self.nmg) - self.gamma*y(0)
        ]
        self.generate_data()
        self.split_data()

    def generate_data(self):
        np.random.seed(self.seed_id)
        from jitcdde import jitcdde_lyap
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
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test  = torch.arange(self.traintime_pts, self.maxtime_pts-1)

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
    lookback_window,
    forecasting_horizon,
    num_bins,
    test_size,
    offset=0,
    MSE=False
):
    x = np.array([point[0] for point in data])  # shape: (N, (1,1))
    y = np.array([point[1] for point in data])  # shape: (N, (1,))

    x_processed = []
    y_processed = []

    # Build sequences
    for i in range(len(x) - lookback_window - forecasting_horizon + 1):
        x_window = x[i : i + lookback_window]
        y_value  = y[i + lookback_window + forecasting_horizon - 1]
        x_processed.append(x_window)
        y_processed.append(y_value)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        x_processed, y_processed, test_size=test_size, shuffle=False
    )

    # If classification, bin the labels
    if not MSE:
        bin_edges = np.linspace(np.min(y), np.max(y), num_bins - 1)
        y_train = np.digitize(y_train, bin_edges)
        y_test  = np.digitize(y_test, bin_edges)

    train_data = []
    for i in range(offset, len(X_train)):
        x_squeezed = np.squeeze(X_train[i], axis=-1) if X_train[i].ndim > 2 else X_train[i]
        train_data.append((x_squeezed, y_train[i]))

    test_data = []
    for i in range(offset, len(X_test)):
        x_squeezed = np.squeeze(X_test[i], axis=-1) if X_test[i].ndim > 2 else X_test[i]
        test_data.append((x_squeezed, y_test[i]))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


###############################################
# 3) Define a Simple RNN Model for Regression
###############################################
class SimpleRNN(nn.Module):
    """
    A simple RNN with one RNN layer + linear output for time-series forecasting.
    """
    def __init__(self, input_size=1, hidden_size=16, output_size=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # take the last timestep
        out = self.fc(out)
        return out


###########################################
# 4) Training, Evaluation, and Forecasting
###########################################
def train_rnn(model, train_loader, num_epochs=50, lr=1e-3):
    """
    Train the RNN model for more epochs (default 50).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            optimizer.zero_grad()
            outputs = model(x_batch)  # shape: (batch_size, 1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")


def evaluate_rnn(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()

            preds.append(output.numpy().flatten())
            targets.append(y_batch.numpy().flatten())

    avg_loss = total_loss / len(test_loader)
    print(f"Test MSE Loss: {avg_loss:.6f}")

    preds   = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return preds, targets, avg_loss


def forecast(model, init_sequence, steps=5):
    """
    Forecast `steps` future values given an initial sequence (lookback_window x 1).
    """
    model.eval()
    forecasts = []

    current_seq = init_sequence.clone().float().unsqueeze(0)  # shape: (1, lookback_window, 1)
    with torch.no_grad():
        for _ in range(steps):
            out = model(current_seq)           # shape: (1,1)
            next_val = out[:, 0].item()        # scalar
            forecasts.append(next_val)

            # shift left by 1 and append the prediction
            new_seq = torch.cat([current_seq[:, 1:, :], out.unsqueeze(2)], dim=1)
            current_seq = new_seq

    return forecasts


#####################################################
# 5) Example usage
#####################################################
if __name__ == "__main__":
    # A) Instantiate Mackey-Glass for a small time range
    dataset = MackeyGlass(
        tau=17,
        constant_past=1.2,
        splits=(30., 10.),  # 30 steps train, 10 steps test
        seed_id=42
    )

    print("\n--- Mackey-Glass Full Time Series (first 10 points) ---")
    print(dataset.mackeyglass_soln[:10])

    # B) Convert to list of (sample, target) pairs
    data_list = []
    for i in range(len(dataset)):
        sample, target = dataset[i]
        sample_np = sample.numpy()  # shape = (1,1)
        target_np = target.numpy()  # shape = (1,)
        data_list.append((sample_np, target_np))

    # C) Create DataLoaders with lookback=3, horizon=1
    lookback_window    = 3
    forecasting_horizon= 1
    num_bins           = 5
    test_size          = 0.2

    train_loader, test_loader = create_time_series_dataset(
        data_list,
        lookback_window,
        forecasting_horizon,
        num_bins,
        test_size,
        offset=0,
        MSE=True
    )

    # D) Print out a few batches from the train_loader
    print("\n--- A few batches from the train_loader (first 10) ---")
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {idx}:")
        print(f"  X_batch shape: {x_batch.shape}, X_batch =\n{x_batch}")
        print(f"  Y_batch shape: {y_batch.shape}, Y_batch =\n{y_batch}")
        if idx >= 9:
            break

    # E) Print out a few batches from the test_loader
    print("\n--- A few batches from the test_loader (first 5) ---")
    for idx, (x_batch, y_batch) in enumerate(test_loader):
        print(f"Test Batch {idx}:")
        print(f"  X_batch shape: {x_batch.shape}, X_batch =\n{x_batch}")
        print(f"  Y_batch shape: {y_batch.shape}, Y_batch =\n{y_batch}")
        if idx >= 4:
            break

    # F) Define RNN model, train for more epochs (e.g. 50), and evaluate
    model = SimpleRNN(input_size=1, hidden_size=16, output_size=1, num_layers=1)
    train_rnn(model, train_loader, num_epochs=50, lr=1e-3)
    preds, targets, test_loss = evaluate_rnn(model, test_loader)

    print("\nSample Predictions vs Targets (Test Set)")
    for i in range(min(5, len(preds))):
        print(f"Pred: {preds[i]:.4f}, Target: {targets[i]:.4f}")

    # G) Forecast 5 steps ahead from the last training batch
    last_train_batch = list(train_loader)[-1]    # (x_batch, y_batch)
    init_seq = last_train_batch[0].squeeze(0)    # shape: (lookback_window, 1)

    future_steps = 5
    future_preds = forecast(model, init_seq, steps=future_steps)

    print(f"\nForecasting {future_steps} steps ahead from the last train window:")
    print(future_preds)
