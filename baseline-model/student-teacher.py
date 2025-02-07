import os
import torch
from utils import MackeyGlass, create_time_series_dataset, RNN, train_rnn, evaluate_rnn, calibrate_uncertainty
from visualize import plot_results, plot_uncertainty

MODEL_PATH_TEMPLATE = "baseline-model/teacher_model_lb{lb}.pth"

def run_experiments():
    tau = 17
    constant_past = 1.2
    splits = (1000., 200.)  # 1000 train, 200 test
    seed_id = 42

    lookback_windows = [3, 5, 7]  # Three different configurations
    forecasting_horizons = [1]  # Next-step forecasting only
    num_epochs = 100
    lr = 1e-3
    hidden_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MackeyGlass(
        tau=tau,
        constant_past=constant_past,
        splits=splits,
        seed_id=seed_id
    )

    data_list = [(sample.numpy()[0, 0], target.numpy()[0]) for sample, target in dataset]

    for lb in lookback_windows:
        teacher_model_path = MODEL_PATH_TEMPLATE.format(lb=lb)


        teacher_train_loader, teacher_test_loader = create_time_series_dataset(
            data_list,
            dataset.ind_train,
            dataset.ind_test,
            lookback_window=lb,
            forecasting_horizon=1, #constant 1
            num_bins=5,
            MSE=True
        )

        teacher_model = RNN(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=3)
        teacher_model.to(device)

        #train/load model
        if os.path.exists(teacher_model_path):
            print(f"\nLoading the saved model for LB={lb} from {teacher_model_path}...")
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            print(f"\nTraining Teacher Model (LB={lb}, FH={1})...")
            train_rnn(teacher_model, teacher_train_loader, num_epochs=num_epochs, lr=lr, device=device)
            print(f"Saving the trained model for LB={lb} to {teacher_model_path}...")
            torch.save(teacher_model.state_dict(), teacher_model_path)

        teacher_model.eval()  
        test_loss, preds, true_values = evaluate_rnn(teacher_model, teacher_test_loader, device=device, verbose=False)

        print(f"\n=== Teacher Model Test Loss ): {test_loss:.6f} ===")

    # call the mg function once and download it.
    # within the create_time_Series_dataset, just give it the pkl data
    #train student
    student_horizons = 10 #put this in a loop later
    student_train_loader, student_test_loader = create_time_series_dataset(
        data_list,
        dataset.ind_train,
        dataset.ind_test,
        lookback_window=lb,
        forecasting_horizon=1, #constant 1
        num_bins=5,
        MSE=True
    )

    helper_train_loader, _ = create_time_series_dataset( #need something about offset here
        data_list,
        dataset.ind_train,
        dataset.ind_test,
        lookback_window=lb,
        forecasting_horizon=1, #constant 1
        num_bins=5,
        # student horizon - 1
        MSE=True
    )
    total_preds = []
    teacher_model.eval()


    for inputs, targets in helper_train_loader:
        inputs = inputs.float().to(device)
        targets = targets.float().to(device).unsqueeze(1)

        with torch.no_grad():
            outputs = teacher_model(inputs)
            preds.append(outputs)
            true_vals.append(targets)

    preds = torch.cat(preds, dim=0)  # Convert list of tensors to a single tensor
    true_vals = torch.cat(true_vals, dim=0)  # Ensure all targets are used

    # Calibrate after collecting all predictions
    calibrated_outputs = calibrate_uncertainty(preds, true_vals)

    total_preds.append(calibrated_outputs)
    

    
if __name__ == "__main__":
    # Ensure the folder exists before saving models
    run_experiments()



