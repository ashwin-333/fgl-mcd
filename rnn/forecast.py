import torch
import pickle 
from dataset import create_time_series_dataset
from model import RNN, mc_dropout_inference, extract_means_vars

def forecast_mcd(lookback_window, teacher_horizon, student_horizon, hidden_size, learning_rate, epochs, optimizer_type):
    print(f'Forecast - MC Dropout - LB: {lookback_window} SH: {student_horizon}')
    device = torch.device("cpu") #mac

    #config
    num_bins = 1
    num_layers = 3 
    num_epochs = epochs

    test_split = 0.25
    dropout_reps = 25

    with open('mg_data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)

    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=test_split,
        MSE=True)

    teacher = RNN(lookback_window, hidden_size, num_bins, num_layers).to(device)
    optimizer = optimizer_type(teacher.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train:
            inputs = inputs.float().to(device).reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = teacher(inputs)
            loss = MSELoss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train):.4f}')

    #student train test
    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        test_size=test_split,
        MSE=True)

    #teacher helper
    train_mc, _ = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=teacher_horizon,
        num_bins=num_bins,
        test_size=test_split,
        offset=student_horizon-teacher_horizon)
    
    means, normalized_vars = extract_means_vars(train_mc, teacher, lookback_window, device, dropout_reps)

    Student = RNN(lookback_window, hidden_size, num_bins, num_layers).to(device)
    optimizer = optimizer_type(Student.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for (inputs, targets), _, var in zip(train, means, normalized_vars):
            inputs = inputs.float().to(device).reshape(1,1,lookback_window)
            targets = targets.float().to(device)
            outputs = Student(inputs)
            loss = MSELoss(outputs, targets)/(var+1e-6)
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

    #config
    hidden_size = 128
    num_bins = 1
    num_layers = 3

    learning_rate = 0.01 
    num_epochs = 20
    test_split = 0.25

    with open('mg_data.pkl', 'rb') as f:
        mackey_glass_data = pickle.load(f)

    train, test = create_time_series_dataset(
        data=mackey_glass_data, 
        lookback_window=lookback_window,
        forecasting_horizon=horizon,
        num_bins=num_bins,
        test_size=test_split,
        MSE=True)


    Model = RNN(lookback_window, hidden_size, num_bins, num_layers).to(device)
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