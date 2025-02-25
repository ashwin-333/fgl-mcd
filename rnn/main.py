from visualize import plot_forecasting_loss, plot_all_forecasting_losses
from forecast import forecast_baseline, forecast_mcd
from dataset import MackeyGlass
from torch.utils.data import Subset
import pickle
import os
import torch
import itertools
from multiprocessing import Pool

def create_data():
    data_path = "mg_data.pkl"
    data_list = None

    if not os.path.exists(data_path):
        mackeyglass = MackeyGlass(tau=17,
            constant_past=1.2,
            nmg = 10,
            beta = 0.2,
            gamma = 0.1,
            dt=1.0,
            splits=(1000., 0.),
            start_offset=0.,
            seed_id=42)
        all_data = Subset(mackeyglass, mackeyglass.ind_train)
        data_list = list(all_data)
        with open(data_path, 'wb') as f:
            pickle.dump(data_list, f)

def test_data():
    with open('data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)
    for i, (inputs, targets) in enumerate(iter(mackey_glass_data)):
        if i == 10:
            break
        print(inputs.item(), targets.item())

def save_baseline_data(baseline_path, baseline_data_dict):
    """ Save baseline data to a JSON file. """
    with open(baseline_path, "wb") as f:
        pickle.dump(baseline_data_dict, f)

def load_baseline_data(baseline_path):
    """ Load baseline data from a JSON file if it exists. """
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            return pickle.load(f)
    return None  # Return None if the file doesn't exist

def run_single_experiment(args):
    hs, lr, num_epochs, optimizer, lookback_window, horizons, teacher_horizon = args
    config = {"hs": hs, "lr": lr, "num_epochs": num_epochs, "optimizer": optimizer.__name__, "lookback_window": lookback_window}
    mcd_data_dict = {str(h): forecast_mcd(lookback_window, teacher_horizon, h, hs, lr, num_epochs, optimizer) for h in horizons}
    return config, mcd_data_dict

def hyperparameter_sweep(horizons, teacher_horizon, hs_values, lr_values, num_epochs_values, optimizer_values, lookback_windows):
    all_results = {}
    param_combinations = list(itertools.product(hs_values, lr_values, num_epochs_values, optimizer_values, lookback_windows))
    
    with Pool(processes=4) as pool:  # Adjust process count based on available CPU cores
        results = pool.map(run_single_experiment, [(hs, lr, num_epochs, optimizer, lookback_window, horizons, teacher_horizon) for hs, lr, num_epochs, optimizer, lookback_window in param_combinations])
    
    all_results = {f"{idx:04d}": result for idx, result in enumerate(results)}
    
    with open("all_results.pth", "wb") as f:
        pickle.dump(all_results, f)
    
    
    
    

        

if __name__ == "__main__":

    create_data()
    #test_data()
    horizons = [i for i in range(2, 26)]
    lookback_window = 10
    teacher_horizon = 1

    

    baseline_path = "baseline_loss.pth"
    all_results_path = "all_results4.pth"

    """
    baseline_data_dict = load_baseline_data(baseline_path)
    if baseline_data_dict is None:
        baseline_data_dict = dict()
        for student_horizon in horizons:
            baseline_data_dict[str(student_horizon)] = forecast_baseline(lookback_window, student_horizon)
        save_baseline_data(baseline_path, baseline_data_dict)
    """
    """
    all_results = None
    with open(all_results_path, "rb") as f:
        all_results = pickle.load(f)

    plot_all_forecasting_losses(all_results, baseline_data_dict, horizons)
    """
    

    hs = [128] #64 and 256 p good
    lr = [0.01]
    num_epochs = [20]
    optimizer = [torch.optim.SGD]
    mcd_lookback_window = [10,20,30]

    baseline_data_dict = dict()
    for horizon in horizons:
        baseline_data_dict[str(horizon)] = forecast_baseline(lookback_window, horizon)

    hyperparameter_sweep(horizons, teacher_horizon, hs, lr, num_epochs, optimizer, mcd_lookback_window)

    all_results = None
    with open(all_results_path, "rb") as f:
        all_results = pickle.load(f)

    plot_all_forecasting_losses(all_results, baseline_data_dict, horizons)


    #mcd_data_dict = dict()
    #for student_horizon in horizons:
    #    mcd_data_dict[str(student_horizon)] = forecast_mcd(lookback_window, teacher_horizon, student_horizon)
    #plot_forecasting_loss(mcd_data_dict, baseline_data_dict)
