import numpy as np
import torch
from jitcdde import jitcdde_lyap, y, t
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import warnings

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")

############################################
# 1) Mackey-Glass Dataset Class
############################################
class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task. """
    def __init__(self, tau, constant_past, nmg=10, beta=0.2, gamma=0.1, dt=1.0, splits=(1000., 200.), start_offset=0., seed_id=42):
        super().__init__()
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.traintime = splits[0]
        self.testtime = splits[1]
        self.start_offset = start_offset
        self.seed_id = seed_id
        self.maxtime = self.traintime + self.testtime + self.dt

        self.traintime_pts = round(self.traintime / self.dt)
        self.testtime_pts = round(self.testtime / self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1

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
def create_time_series_dataset(data, train_indices, test_indices, lookback_window, forecasting_horizon, num_bins, MSE=False):
    """
    Create training and testing datasets based on predefined indices.
    """
    x = np.array([point[0] for point in data])  # shape: (N,)
    y = np.array([point[1] for point in data])  # shape: (N,)

    x_processed_train = []
    y_processed_train = []

    x_processed_test = []
    y_processed_test = []

    def is_valid_sequence(start_idx, end_idx, valid_indices):
        return all(idx in valid_indices for idx in range(start_idx, end_idx))

    for i in train_indices:
        start = i - lookback_window - forecasting_horizon + 1
        end = i
        if start < 0:
            continue
        target_idx = i + forecasting_horizon - 1
        if target_idx >= len(y):
            continue
        if is_valid_sequence(start, end, train_indices):
            x_window = x[start : start + lookback_window]
            y_value  = y[target_idx]
            x_processed_train.append(x_window)
            y_processed_train.append(y_value)

    for i in test_indices:
        start = i - lookback_window - forecasting_horizon + 1
        end = i
        if start < 0:
            continue
        target_idx = i + forecasting_horizon - 1
        if target_idx >= len(y):
            continue
        if is_valid_sequence(start, end, test_indices):
            x_window = x[start : start + lookback_window]
            y_value = y[target_idx]
            x_processed_test.append(x_window)
            y_processed_test.append(y_value)

    X_train = np.array(x_processed_train)
    y_train = np.array(y_processed_train)
    X_test = np.array(x_processed_test)
    y_test = np.array(y_processed_test)

    train_data = [(torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(-1), torch.tensor(y_train[i], dtype=torch.float32))
                  for i in range(len(X_train))]
    test_data = [(torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(-1), torch.tensor(y_test[i], dtype=torch.float32))
                 for i in range(len(X_test))]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=1, shuffle=False)

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


###########################################
# 4) Training and Evaluation Functions
###########################################
def train_rnn(model, train_loader, num_epochs=50, lr=1e-3, device='cpu'):
    """
    Train the RNN model.
    """
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}")


def evaluate_rnn(model, test_loader, device='cpu', verbose=True):
    """
    Evaluate the RNN model.
    """
    model.eval()
    criterion = torch.nn.GaussianNLLLoss()
    total_loss = 0.0
    predictions = []
    true_values = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device).unsqueeze(1)

            mean = model(x_batch)
            variance = torch.ones_like(mean).to(device)
            loss = criterion(mean, y_batch, variance)
            total_loss += loss.item() * x_batch.size(0)

            predictions.extend(mean.cpu().detach().numpy())
            true_values.extend(y_batch.cpu().detach().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()

    if verbose:
        print(f"Test Gaussian NLL Loss: {avg_loss:.6f}")
    return avg_loss, predictions, true_values

def evaluate_helper(model, test_loader, device='cpu', verbose=True):
    """
    Evaluate the RNN model.
    """
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    predictions = []
    true_values = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device).unsqueeze(1)

            mean = model(x_batch)
            loss = criterion(mean, y_batch)

            predictions.extend(mean.cpu().detach().numpy())
            true_values.extend(y_batch.cpu().detach().numpy())
    
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()

    return predictions, true_values

###########################
# 5) Forecasting Function
###########################

def forecast(model, init_sequence, steps=5, device='cpu'):
    """
    Forecast steps future values given an initial sequence (lookback_window x 1).
    """
    model.eval()
    forecasts = []

    current_seq = init_sequence.clone().float().to(device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            out = model(current_seq)
            next_val = out[:, 0].item()
            forecasts.append(next_val)
            next_input = torch.tensor([[next_val]], dtype=torch.float32).to(device)
            current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(0)], dim=1)

    return forecasts

##############################
# 6) Calculating uncertainty
##############################
def calibrate_uncertainty(preds, true_values):
    preds_np = preds
    true_values_np = true_values
    
    ir = IsotonicRegression(out_of_bounds="clip")
    calibrated_preds = ir.fit_transform(preds_np, true_values_np)
    #calibrated_preds = np.abs(calibrated_preds - true_values) * 2


    return torch.tensor(calibrated_preds, dtype=torch.float32, device=preds.device)

"""def calibrate_uncertainty(preds, true_values):
    alpha = 2
    
    ir = IsotonicRegression(out_of_bounds="clip")
    calibrated_preds = ir.fit_transform(preds, true_values)
    return calibrated_preds

    residuals = true_values - calibrated_preds


    positive_residuals = np.maximum(residuals, 0)
    negative_residuals = np.minimum(residuals, 0)

    upper_residual = np.percentile(positive_residuals, 95)  # 95th percentile
    lower_residual = np.percentile(negative_residuals, 5)   # 5th percentile

    ir_upper = calibrated_preds + alpha * upper_residual  # Upper bound
    ir_lower = calibrated_preds + alpha * lower_residual  # Lower bound
    

    return calibrated_preds, ir_lower,  ir_upper
"""
