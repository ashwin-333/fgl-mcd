import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMModel:
    """
    A generalizable LSTM-based time series forecasting model.
    The architecture is defined directly in the constructor for consistency with other models.
    """
    def __init__(self, input_timesteps, input_dim, lstm_units_1=50, lstm_units_2=50,
                 dropout_rate=0.2, dense_units=1, loss='mse', optimizer='adam'):
        self.model = Sequential()

        # First LSTM layer with return_sequences=True to stack another LSTM layer.
        self.model.add(LSTM(lstm_units_1, input_shape=(input_timesteps, input_dim), return_sequences=True))
        # Dropout layer.
        self.model.add(Dropout(dropout_rate))
        # Second LSTM layer with return_sequences=False for final output.
        self.model.add(LSTM(lstm_units_2, return_sequences=False))
        # Final Dense layer for prediction.
        self.model.add(Dense(dense_units, activation='linear'))

        self.model.compile(loss=loss, optimizer=optimizer)
        print('[LSTMModel] Model Compiled')

    def load_model(self, filepath):
        print(f'[LSTMModel] Loading model from file {filepath}')
        self.model = load_model(filepath)

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print(f'[LSTMModel] Training Started - {epochs} epochs, {batch_size} batch size')

        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print(f'[LSTMModel] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print(f'[LSTMModel] Training Started - {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')

        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.h5')
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print(f'[LSTMModel] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data (1 step ahead predictions)
        print('[LSTMModel] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict a sequence of prediction_len steps before shifting the window forward.
        print('[LSTMModel] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by one new prediction each time, re-run predictions on the new window.
        print('[LSTMModel] Predicting Sequence Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted


def mc_dropout_inference(model_instance, X, num_reps=100):
    """
    Perform Monte Carlo dropout inference by making multiple stochastic forward passes.
    
    Args:
        model_instance: an instance of LSTMModel.
        X: input tensor.
        num_reps: number of repetitions.
    
    Returns:
        mean_preds: mean predictions over the repetitions.
        var_preds: variance of predictions.
    """
    # Enable dropout by setting the model to train mode.
    model_instance.model.train()
    preds = []
    for _ in range(num_reps):
        with np.errstate(all='ignore'):
            output = model_instance.model.predict(X)
        preds.append(output)
    preds = np.stack(preds)
    mean_preds = np.mean(preds, axis=0)
    var_preds = np.var(preds, axis=0)
    return mean_preds, var_preds


def extract_means_vars(train_loader, teacher, lookback_window, device, dropout_reps):
    """
    Extract mean predictions and normalized variances from the teacher model using MC dropout.
    
    Args:
        train_loader: data loader for the training set.
        teacher: an instance of LSTMModel.
        lookback_window: length of the input sequence.
        device: device to run on.
        dropout_reps: number of MC dropout repetitions.
    
    Returns:
        means: stacked mean predictions.
        normalized_vars: normalized variances.
    """
    means = []
    vars = []
    teacher.model.evaluate  # ensure model is built
    for inputs, _ in train_loader:
        inputs = inputs.astype('float32').reshape(1, lookback_window, -1)
        mean_preds, var_preds = mc_dropout_inference(teacher, inputs, dropout_reps)
        means.append(mean_preds)
        vars.append(var_preds)
    means = np.stack(means)
    vars = np.stack(vars)

    normalized_vars = 0.1 + (vars - vars.min()) / (vars.max() - vars.min()) * (1 - 0.1)
    return means, normalized_vars


# # Example usage:
# if __name__ == '__main__':
#     # Set parameters for the model.
#     input_timesteps = 10   # lookback window
#     input_dim = 1          # number of input features per time step
#     save_directory = './'

#     # Instantiate the LSTM model directly without a configuration dictionary.
#     lstm_model = LSTMModel(input_timesteps, input_dim, lstm_units_1=50, lstm_units_2=50,
#                            dropout_rate=0.2, dense_units=1, loss='mse', optimizer='adam')

#     # Create a dummy dataset.
#     x_dummy = np.random.randn(100, input_timesteps, input_dim)
#     y_dummy = np.random.randn(100, 1)

#     # Train the model.
#     lstm_model.train(x_dummy, y_dummy, epochs=5, batch_size=16, save_dir=save_directory)

#     # Make a prediction.
#     predictions = lstm_model.predict_point_by_point(x_dummy)
#     print("Predictions shape:", predictions.shape)
