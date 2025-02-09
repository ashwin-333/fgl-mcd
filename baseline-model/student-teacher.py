import os
import torch
from utils import MackeyGlass, create_time_series_dataset, RNN, train_rnn, evaluate_rnn, calibrate_uncertainty, evaluate_helper
from visualize import plot_results, plot_uncertainty
import torch.optim as optim
import numpy as np

MODEL_PATH_TEMPLATE = "baseline-model/teacher_model_lb{lb}.pth"

def run_experiments():
    tau = 17
    constant_past = 1.2
    splits = (1000., 200.)  # 1000 train, 200 test
    seed_id = 42

    lookback_windows = [3, 5, 7]  # Three different configurations
    forecasting_horizons = [1]  # Next-step forecasting only
    num_epochs = 20
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
        lookback_window=3,
        forecasting_horizon=student_horizons, #constant 1
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

    

    helper_preds, helper_true_vals = evaluate_helper(teacher_model, helper_train_loader, device=device, verbose=False)
    vars = calibrate_uncertainty(helper_preds, helper_true_vals)

    var_min = vars.min()
    var_max = vars.max() 

    vars = 0.1 + (vars - var_min) / (var_max - var_min) * (1 - 0.1)

    vars = vars[student_horizons-1:]
    vars = vars.to(device)

    student_model = RNN(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=3)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    student_model.to(device)

#train student
    for epoch in range(num_epochs):
        total_loss = 0
        print(epoch)
        for (inputs, targets), var in zip(student_train_loader, vars):
            var = var.float().to(device)
            var = var.unsqueeze(0).unsqueeze(0)
            inputs = inputs.float().to(device)
            targets = targets.float().to(device).unsqueeze(1)
            outputs = student_model(inputs)
            loss = torch.nn.GaussianNLLLoss()
            output_loss = loss(outputs, targets, var)
            optimizer.zero_grad()
            output_loss.backward()  
            optimizer.step()
            total_loss += output_loss.item()
            print(output_loss.item())
    print(f"Loss Average: {total_loss/len(student_train_loader.dataset)}")


    

    
if __name__ == "__main__":
    # Ensure the folder exists before saving models
    run_experiments()



