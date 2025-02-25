import torch
import pickle 
from dataset import create_time_series_dataset
from model import RNN, mc_dropout_inference

def forecast_mcd(lookback_window, teacher_horizon, student_horizon, hs, lr, epochs, optimizer_type):
    print(f'Forecast - MC Dropout - LB: {lookback_window} SH: {student_horizon}')
    device = torch.device("cpu") #mac

    #hyperparameters
    num_bins = 1 # this is how lenient we are with the network prediction
    input_size = lookback_window  # one value 
    hidden_size = hs # hidden neurons in RNN
    output_size = num_bins # output neurons = num_bins
    num_layers = 3 # RNN layers
    batch_size = 1 # must be 1 for continual learning!!!
    learning_rate = lr 
    num_epochs = epochs
    test_split = 0.25
    dropout_reps = 25

    with open('data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)

    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=test_split,
        MSE=True)

    Teacher = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = optimizer_type(Teacher.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device).reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Teacher(inputs)
            loss = MSELoss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')

    ###Training the Student###

    #student training and testing set
    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        test_size=test_split,
        MSE=True)

    #helper for teacher to extract PDFs 
    train_mc, _ = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=test_split,
        offset=student_horizon-teacher_horizon)

    means = []
    vars = []
    Teacher.eval()
    for inputs, targets in train_mc:
        inputs = inputs.float().to(device).reshape(1, 1, lookback_window)
        #targets = targets.float()

        # Perform MC Dropout inference on each test sample
        mean_preds, var_preds = mc_dropout_inference(Teacher, inputs, dropout_reps)
        means.append(mean_preds)
        vars.append(var_preds)  # Ensure variance is 1D

    means = torch.stack(means).to(device)
    vars = torch.stack(vars).to(device)

    # Apply min-max normalization on variances to normalize them between 0.5 and 2
    var_min = vars.min()  # Find minimum variance
    var_max = vars.max()  # Find maximum variance

    # Normalize the variance
    nvars = 0.1 + (vars - var_min) / (var_max - var_min) * (1 - 0.1)

    Student = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = optimizer_type(Student.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for (inputs, targets), mean, var in zip(train, means, nvars):
            inputs = inputs.float().to(device).reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Student(inputs)
            loss = MSELoss(outputs, targets)/(var+1e-4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')
    MAELoss = torch.nn.L1Loss()
    Student.eval()
    with torch.no_grad():
        msetotal_loss = 0
        maetotal_loss = 0
        predictions = []
        true_values = []
        for inputs, targets in test:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Student(inputs)
            mseloss = MSELoss(outputs, targets)
            msetotal_loss += mseloss.item()
            maeloss = MAELoss(outputs, targets)
            maetotal_loss += maeloss.item()
            predictions.append(outputs[0].cpu().numpy())
            true_values.append(targets[0].cpu().detach().numpy())
        print(f'MSE Loss: {msetotal_loss/len(test):.4f}')
        print(f'MAE Loss: {maetotal_loss/len(test):.4f}\n')
    return (msetotal_loss/len(test), maetotal_loss/len(test))

def forecast_baseline(lookback_window, horizon):
    print(f'Forecast- Baseline Model - LB: {lookback_window} H: {horizon}')
    device = torch.device("cpu") #mac

    #hyperparameters
    input_size = lookback_window  # one value 
    hidden_size = 32 # hidden neurons in RNN
    output_size = 1
    num_layers = 3 # RNN layers
    learning_rate = 0.001 
    num_epochs = 5
    test_split = 0.25

    with open('data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)

    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=horizon,
        num_bins=1,
        test_size=test_split,
        MSE=True)


    Model = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device).reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Model(inputs)
            loss = MSELoss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')
    
    MAELoss = torch.nn.L1Loss()
    Model.eval()
    with torch.no_grad():
        msetotal_loss = 0
        maetotal_loss = 0
        predictions = []
        true_values = []
        for inputs, targets in test:
            inputs = inputs.float().to(device)
            inputs = inputs.reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Model(inputs)
            mseloss = MSELoss(outputs, targets)
            msetotal_loss += mseloss.item()
            maeloss = MAELoss(outputs, targets)
            maetotal_loss += maeloss.item()
            predictions.append(outputs[0].cpu().numpy())
            true_values.append(targets[0].cpu().detach().numpy())
        print(f'MSE Loss: {msetotal_loss/len(test):.4f}')
        print(f'MAE Loss: {maetotal_loss/len(test):.4f}\n')
    return (msetotal_loss/len(test), maetotal_loss/len(test))