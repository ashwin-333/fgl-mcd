import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=3, p_dropout=0.1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out

def mc_dropout_inference(model, X, num_reps=100):
    model.train() 
    preds = []

    for _ in range(num_reps):
        with torch.no_grad():
            outputs = model(X)
        preds.append(outputs)

    preds = torch.stack(preds)
    mean_preds = torch.mean(preds, dim=0)
    var_preds = torch.var(preds, dim=0)
    return mean_preds, var_preds

def extract_means_vars(train_mc, teacher, lookback_window, device, dropout_reps):
    means = []
    vars = []
    teacher.eval()
    for inputs, _ in train_mc:
        inputs = inputs.float().to(device).reshape(1, 1, lookback_window)

        mean_preds, var_preds = mc_dropout_inference(teacher, inputs, dropout_reps)
        means.append(mean_preds)
        vars.append(var_preds)

    means = torch.stack(means).to(device)
    vars = torch.stack(vars).to(device)

    normalized_vars = 0.1 + (vars - vars.min()) / (vars.max() - vars.min()) * (1 - 0.1)
    return means, normalized_vars