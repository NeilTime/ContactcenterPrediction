####################################################
# lstm_model.py
#
# Updated script with enhanced model complexity
####################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plot_functions import (
    plot_actual_vs_predicted,
    plot_error_distributions,
    plot_highest_and_lowest_error_days,
    plot_monthly_predicted_vs_actual
)

# Quantile Loss Function (Consider switching to MSE or MAE)
class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        error = target - preds
        loss = torch.max((self.quantile * error), ((self.quantile - 1) * error))
        return torch.mean(loss)

# PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (samples, time_steps, features)
        self.y = torch.tensor(y, dtype=torch.float32)  # (samples, )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Self-Attention Layer with Enhanced Regularization
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))  # Residual connection with dropout
        return self.layer_norm2(x)  # Additional LayerNorm if needed

# LSTM Model with Self-Attention and Enhanced Regularization
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.3, add_noise=True, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.add_noise = add_noise
        if self.add_noise:
            self.noise = GaussianNoise(0.05)  # Adjusted noise level

        # Bidirectional LSTM
        # self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=bidirectional)
        # self.dropout1 = nn.Dropout(dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=bidirectional)
        # self.dropout2 = nn.Dropout(dropout)
        # self.attention = SelfAttentionLayer(embed_dim=hidden_dim * 2 if bidirectional else hidden_dim, num_heads=num_heads)
        # self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

        # self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=bidirectional)
        # self.dropout1 = nn.Dropout(dropout)
        # self.lstm2 = nn.LSTM(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=bidirectional)
        # self.dropout2 = nn.Dropout(dropout)
        # self.attention = SelfAttentionLayer(embed_dim=hidden_dim * 2 if bidirectional else hidden_dim, num_heads=num_heads)
        # self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        
        # Single LSTM Layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1,  # Changed to 1 layer
            batch_first=True, 
            dropout=0.0,  # Dropout not needed with a single layer
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)  # Adjust input features based on bidirectionality

    def forward(self, x):
        if self.add_noise:
            x = self.noise(x)
        # x, _ = self.lstm1(x)
        # x = self.dropout1(x)
        # x, _ = self.lstm2(x)
        # x = self.dropout2(x)
        # # Prepare for attention: transpose to (seq_len, batch, hidden_dim)
        # x = x.permute(1, 0, 2)
        # x = self.attention(x)
        # # Transpose back to (batch, seq_len, hidden_dim)
        # x = x.permute(1, 0, 2)
        # # Use the last time step's output
        # x = x[:, -1, :]
        # x = self.fc(x)
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_dim * num_directions)
        x = self.dropout(x)
        # Use the last time step's output
        x = x[:, -1, :]  # Shape: (batch, hidden_dim * num_directions)
        x = self.fc(x)  # Shape: (batch, 1)
        return x

        return x

# Custom Gaussian Noise Layer with Increased Noise
class GaussianNoise(nn.Module):

    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

# DataPreprocessor and ModelEvaluator remain as classes within this module.
# Ensure their methods are updated with appropriate import paths as shown earlier.
class DataLoaderCustom:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load the data from the specified file path
        time_series_df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        time_series_df.sort_values(by='timestamp', inplace=True)
        return time_series_df

class DataPreprocessorCustom:
    def __init__(self, hours=9, weekdays=5):
        self.time_steps = hours * weekdays
        self.scaler = StandardScaler()

    def add_lag_features(self, df):
        # Create lag features based on the previous week (7 days before)
        df['calls_lag_1week'] = df['hourly_calls'].shift(self.time_steps)
        # df['avg_talk_time_lag_1week'] = df['avg_talk_time'].shift(self.time_steps)
        
        # Drop rows with missing lag values (first 7 days will have NaNs)
        df.dropna(inplace=True)
        return df
    
    def add_date_features(self, df):
        # Extract year, month, and day for seasonality
        df['year'] = df['timestamp'].dt.year
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # Add cyclical encoding for day of the year
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

        # Convert hour to a 0-8 scale for workday hours (8 to 16)
        df['workday_hour'] = df['timestamp'].dt.hour - 8

        # Cyclical encoding for workday hours (0 to 8)
        df['workday_hour_sin'] = np.sin(2 * np.pi * df['workday_hour'] / 9)
        df['workday_hour_cos'] = np.cos(2 * np.pi * df['workday_hour'] / 9)

        return df

    def normalize_data(self, df):
        # Apply log transformation to 'hourly_calls' for normalization
        df['hourly_calls'] = np.log1p(df['hourly_calls'])  # log(1 + x) to avoid log(0) issues
        df['avg_talk_time'] = pd.to_timedelta(df['avg_talk_time']).dt.total_seconds()
        # df['avg_talk_time_lag_1week'] = pd.to_timedelta(df['avg_talk_time_lag_1week']).dt.total_seconds()

        # Define the columns in the exact order
        columns = [
            'hourly_calls',         # Target feature for prediction (must be first)
            'year',                 # Year for seasonality
            'day_of_year_sin',      # Seasonal pattern (day of the year) - cyclic
            'day_of_year_cos',      # Seasonal pattern (day of the year) - cyclic
            'workday_hour_sin',     # Hour of the workday - cyclic
            'workday_hour_cos'      # Hour of the workday - cyclic
        ]

        scaled_data = self.scaler.fit_transform(df[columns])

        return pd.DataFrame(scaled_data, columns=columns, index=df.index)
    
    def create_sequences(self, data):
        x, y = [], []
        for i in range(len(data) - self.time_steps):
            x.append(data[i:i + self.time_steps])
            y.append(data[i + self.time_steps, 0])  # Predicting 'hourly_calls'
        return np.array(x), np.array(y)
    
    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, shuffle=False)

class ModelEvaluatorCustom:
    def __init__(self, scaler):
        self.scaler = scaler

    def inverse_transform(self, predictions, actual):
        """
        Manually inverse transform the 'hourly_calls' using the scaler's parameters
        and reverse the log1p transformation.

        Args:
            predictions (numpy.ndarray): The model's predictions.
            actual (numpy.ndarray): The actual target values.

        Returns:
            tuple: Inversely transformed predicted and actual calls in original scale.
        """
        # Extract the scaler's parameters for 'hourly_calls' (first feature)
        scale_hourly_calls = self.scaler.scale_[0]
        mean_hourly_calls = self.scaler.mean_[0]

        # Perform the inverse scaling
        predicted_scaled = predictions * scale_hourly_calls + mean_hourly_calls
        actual_scaled = actual * scale_hourly_calls + mean_hourly_calls

        # Reverse the log1p transformation
        predicted_calls = np.expm1(predicted_scaled)
        actual_calls = np.expm1(actual_scaled)

        return predicted_calls, actual_calls

    def plot_results(self, y_test_actual, predicted_calls):
        plot_actual_vs_predicted(y_test_actual, predicted_calls)

    def calculate_average_error(self, data):
        # Hourly error calculation
        data['hourly_error'] = np.abs(data['actual_calls'] - data['predicted_calls']) / data['actual_calls'] * 100
        average_hourly_error = data['hourly_error'].mean()
        print(f"Average Hourly Prediction Error: {average_hourly_error:.2f}%")

        # Daily total error calculation
        data['date'] = data['timestamp'].dt.date  # Extract date for grouping
        daily_totals = data.groupby('date').agg({
            'actual_calls': 'sum',
            'predicted_calls': 'sum'
        }).reset_index()

        # Calculate daily error based on total calls per day
        daily_totals['daily_error'] = np.abs(daily_totals['actual_calls'] - daily_totals['predicted_calls']) / daily_totals['actual_calls'] * 100
        average_daily_error = daily_totals['daily_error'].mean()
        print(f"Average Daily Total Prediction Error: {average_daily_error:.2f}%")

        # Print top 10 max and min errors for hourly and daily predictions
        top_max_hourly_error = data.nlargest(10, 'hourly_error')[['timestamp', 'actual_calls', 'predicted_calls', 'hourly_error']]
        print(f"Top 10 Maximum Hourly Errors:\n{top_max_hourly_error}")

        top_min_hourly_error = data.nsmallest(10, 'hourly_error')[['timestamp', 'actual_calls', 'predicted_calls', 'hourly_error']]
        print(f"Top 10 Minimum Hourly Errors:\n{top_min_hourly_error}")

        top_max_daily_error = daily_totals.nlargest(10, 'daily_error')[['date', 'actual_calls', 'predicted_calls', 'daily_error']]
        print(f"Top 10 Maximum Daily Total Errors:\n{top_max_daily_error}")

        top_min_daily_error = daily_totals.nsmallest(10, 'daily_error')[['date', 'actual_calls', 'predicted_calls', 'daily_error']]
        print(f"Top 10 Minimum Daily Total Errors:\n{top_min_daily_error}")

        # Plotting average error per day in a box plot with std and singular datapoints
        daily_totals['month'] = pd.to_datetime(daily_totals['date']).dt.to_period('M')
        monthly_means = daily_totals.groupby('month')['daily_error'].mean().reset_index()
        plot_error_distributions(daily_totals, monthly_means)

        # Merge daily_error into data based on 'date'
        data = data.merge(daily_totals[['date', 'daily_error']], on='date', how='left')

        # Return daily_totals and updated data for further analysis
        return daily_totals, data

    def evaluate_and_plot(self, predictions, y_test, timestamps):
        # Inverse transform predictions and actual values
        predicted_calls, y_test_actual = self.inverse_transform(predictions, y_test)

        # Create a DataFrame for plotting and error calculation
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'predicted_calls': predicted_calls,
            'actual_calls': y_test_actual
        })

        # Calculate and print error metrics
        daily_totals, data = self.calculate_average_error(data)

        # Plot actual vs predicted calls
        self.plot_results(y_test_actual=y_test_actual, predicted_calls=predicted_calls)

        # **Debugging Statement: Check DataFrame Columns**
        print(f"Data columns before plotting highest and lowest error days: {data.columns.tolist()}")

        # Plot highest and lowest error days
        plot_highest_and_lowest_error_days(data)

        # **New Plot: Monthly Predicted vs Actual Calls**
        plot_monthly_predicted_vs_actual(data)