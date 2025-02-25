import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the Mamba2 module.
from mamba_ssm import Mamba2

class Mamba2Forecast(nn.Module):
    """
    A forecasting model that uses a Mamba2 block (or a stack of them)
    in place of an RNN. It first embeds the input, passes it through the Mamba2 layer(s),
    then applies dropout and two fully connected layers to produce the forecast.
    """
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1, p_dropout=0.1):
        super(Mamba2Forecast, self).__init__()
        
        # Embedding layer to project input features to the model dimension.
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # First Mamba2 block. Here we use default parameters for illustration;
        # you can tune d_state, d_conv, expand, etc., as needed.
        self.mamba2 = Mamba2(
            d_model=hidden_size,
            d_state=hidden_size,    # set state dimension equal to hidden_size for simplicity
            d_conv=4,               # convolution width (default value)
            expand=2,               # expansion factor (default value)
            headdim=hidden_size // 2 if hidden_size >= 2 else hidden_size,
            d_ssm=hidden_size      # SSM dimension
        )
        
        # If multiple layers are desired, stack additional Mamba2 blocks.
        self.num_layers = num_layers
        if num_layers > 1:
            self.mamba_layers = nn.ModuleList([
                Mamba2(
                    d_model=hidden_size,
                    d_state=hidden_size,
                    d_conv=4,
                    expand=2,
                    headdim=hidden_size // 2 if hidden_size >= 2 else hidden_size,
                    d_ssm=hidden_size
                ) for _ in range(num_layers - 1)
            ])
        else:
            self.mamba_layers = None
        
        # Dropout and fully-connected layers (similar to the original RNN model).
        self.dropout = nn.Dropout(p_dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass.
        x: Tensor of shape (batch, seq_len, input_size)
        """
        # Embed the input to the desired hidden size.
        x = self.embedding(x)  # now shape: (batch, seq_len, hidden_size)
        
        # Pass through the first Mamba2 block.
        out = self.mamba2(x)
        
        # If additional layers are defined, pass the output through them sequentially.
        if self.mamba_layers is not None:
            for layer in self.mamba_layers:
                out = layer(out)
        
        # For forecasting, we take the output at the last time step.
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

def mc_dropout_inference(model, X, num_reps=100):
    """
    Perform Monte Carlo dropout inference by making multiple stochastic forward passes.
    
    Args:
        model: the forecasting model.
        X: input tensor.
        num_reps: number of repetitions.
    
    Returns:
        mean_preds: mean predictions over the repetitions.
        var_preds: variance of predictions.
    """
    # Enable dropout by setting the model to train mode.
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

def extract_means_vars(train_loader, teacher, lookback_window, device, dropout_reps):
    """
    Extract mean predictions and normalized variances from the teacher model using MC dropout.
    
    Args:
        train_loader: data loader for the training set.
        teacher: the forecasting model.
        lookback_window: length of the input sequence.
        device: device to run on.
        dropout_reps: number of MC dropout repetitions.
    
    Returns:
        means: stacked mean predictions.
        normalized_vars: normalized variances.
    """
    means = []
    vars = []
    teacher.eval()
    for inputs, _ in train_loader:
        # Reshape inputs to (batch=1, seq_len=lookback_window, input_size=1) if necessary.
        # (Adjust this reshape if your data format differs.)
        inputs = inputs.float().to(device).reshape(1, lookback_window, 1)
        mean_preds, var_preds = mc_dropout_inference(teacher, inputs, dropout_reps)
        means.append(mean_preds)
        vars.append(var_preds)
    means = torch.stack(means).to(device)
    vars = torch.stack(vars).to(device)
    
    # Normalize the variances to a desired range.
    normalized_vars = 0.1 + (vars - vars.min()) / (vars.max() - vars.min()) * (1 - 0.1)
    return means, normalized_vars

# Example usage:
# if __name__ == '__main__':
#     # Set up parameters for time series forecasting.
#     batch_size = 16
#     seq_len = 50       # lookback window
#     input_size = 1     # number of input features per time step
#     hidden_size = 32
#     output_size = 1

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Instantiate the Mamba2-based forecasting model.
#     # Here we use 2 layers (1 initial mamba2 block + 1 additional stacked layer) and a dropout probability of 0.1.
#     model = Mamba2Forecast(input_size, hidden_size, output_size, num_layers=2, p_dropout=0.1).to(device)
    
#     # Create a dummy input tensor with shape (batch_size, seq_len, input_size).
#     x = torch.randn(batch_size, seq_len, input_size).to(device)
    
#     # Forward pass.
#     output = model(x)
#     print("Model output shape:", output.shape)
