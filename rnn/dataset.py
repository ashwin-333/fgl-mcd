import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from jitcdde import jitcdde_lyap, y, t
from sklearn.model_selection import train_test_split
class MackeyGlass(Dataset):

    def __init__(self, tau, constant_past, nmg=10, beta=0.2, gamma=0.1, dt=1.0, splits=(800., 200.), start_offset=0., seed_id=42):
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


def create_time_series_dataset(data, lookback_window, forecasting_horizon, num_bins, test_size, offset=0, MSE=False):
    x = np.array([point[0] for point in data])
    y = np.array([point[1] for point in data])
    x_processed = []
    y_processed = []

    for i in range(len(x) - lookback_window - forecasting_horizon + 1):
        x_window = x[i:i + lookback_window]
        y_value = y[i + lookback_window + forecasting_horizon - 1]
        x_processed.append(x_window)
        y_processed.append(y_value)

    X_train, X_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=test_size, shuffle=False)
    bin_edges = np.linspace(np.min(y), np.max(y), num_bins - 1)
    if not MSE:
        y_train = np.digitize(y_train, bin_edges)
        y_test = np.digitize(y_test, bin_edges)

    train  = [(X_train[i].squeeze(-1), y_train[i]) for i in range(offset, len(X_train))]
    test = [(X_test[i].squeeze(-1), y_test[i]) for i in range(offset, len(X_test))]


    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return train_loader, test_loader 