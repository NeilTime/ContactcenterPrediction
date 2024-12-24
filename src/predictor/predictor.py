####################################################
# predictor.py
#
# Updated script to handle Torch FutureWarning and improved predictions
####################################################

import torch
from src.models.lstm_model import LSTMModel, DataPreprocessorCustom, ModelEvaluatorCustom
from src.data.data_loader import DataLoaderCustom

class Predictor:
    def __init__(self, model_weights_path, data_file_path, input_dim=6, hidden_dim=48, num_layers=2, num_heads=4, dropout=0.3, add_noise=True, quantile=0.7, device='cpu'):
        self.model_weights_path = model_weights_path
        self.data_file_path = data_file_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_noise = add_noise
        self.quantile = quantile
        self.device = device

        self.data_loader = DataLoaderCustom(data_file_path)
        self.data_preprocessor = DataPreprocessorCustom(hours=9, weekdays=5)
        self.model = LSTMModel(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            num_heads=self.num_heads, 
            dropout=self.dropout, 
            add_noise=self.add_noise
        ).to(self.device)

        # Load model weights with weights_only=True
        try:
            state_dict = torch.load(self.model_weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        except TypeError:
            # If weights_only is not supported in your PyTorch version, fallback to weights_only=False
            state_dict = torch.load(self.model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Note: weights_only parameter is not available in your PyTorch version.")

    def load_data_and_prepare_sequences(self):
        df = self.data_loader.load_data()
        df = self.data_preprocessor.add_lag_features(df)
        df = self.data_preprocessor.add_date_features(df)
        processed_df = self.data_preprocessor.normalize_data(df)
        X, y = self.data_preprocessor.create_sequences(processed_df.values)
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        # Extract timestamps corresponding to the sequences
        timestamps = df['timestamp'].values[self.data_preprocessor.time_steps:]
        return X, y, timestamps

    def load_model_and_predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

    def predict_and_plot(self):
        # Load data, prepare sequences
        X_test, y_test, timestamps = self.load_data_and_prepare_sequences()
        # Make predictions
        predictions = self.load_model_and_predict(X_test)
        # Evaluate and plot
        self.evaluate_and_plot(predictions, y_test, timestamps)

    def evaluate_and_plot(self, predictions, y_test, timestamps):
        # Instantiate ModelEvaluator
        model_evaluator = ModelEvaluatorCustom(scaler=self.data_preprocessor.scaler)
        # Evaluate and plot
        model_evaluator.evaluate_and_plot(predictions, y_test, timestamps)