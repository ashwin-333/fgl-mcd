import torch.nn as nn
import torch

class RNN(nn.Module):
    """
    An RNN with three RNN layers and a linear output for time-series forecasting.
    """
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

def mc_dropout_inference(model, X, n_forward_passes=100):
    model.eval() 
    dropout_preds = []

    for _ in range(n_forward_passes):
        model.train()
        with torch.no_grad():
            preds = model(X)
        dropout_preds.append(preds)

    dropout_preds = torch.stack(dropout_preds)
    mean_preds = torch.mean(dropout_preds, axis=0)
    std_preds = torch.var(dropout_preds, axis=0)
    return mean_preds, std_preds