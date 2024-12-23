####################################################
# lstm_model.py
#
# This script contains the classes for loading 
# data, preprocessing data, building an LSTM model,
# and evaluating the model.
####################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.layers import GaussianNoise
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def quantile_loss(quantile):
    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * err, (quantile - 1) * err))
    return loss


# Class 1: DataLoader
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load the data from the specified file path
        time_series_df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        time_series_df.sort_values(by='timestamp', inplace=True)
        return time_series_df


class DataPreprocessor:
    def __init__(self, hours=9, weekdays=5):
        self.time_steps = hours * weekdays
        self.scaler = StandardScaler()

    def add_lag_features(self, df):
        # Create lag features based on the previous week (7 days before)
        df['calls_lag_1week'] = df['hourly_calls'].shift(self.time_steps)
        df['avg_talk_time_lag_1week'] = df['avg_talk_time'].shift(self.time_steps)
        
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
        df['avg_talk_time_lag_1week'] = pd.to_timedelta(df['avg_talk_time_lag_1week']).dt.total_seconds()

        # Scale the data as before
        columns = [
            'hourly_calls',         # Target feature for prediction
            # 'avg_talk_time',        # Average talk time as a feature
            # 'calls_lag_1week',      # Call volume from the same hour one week ago
            # 'avg_talk_time_lag_1week', # Lagged talk time for context
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

# Class 3: LSTMModel
class LSTMModel:
    def __init__(self, input_shape, lr=0.1, add_noise=True):
        self.input_shape = input_shape
        self.model = self.build_model(lr=lr, add_noise=add_noise)

    def build_model(self, lr=0.1, add_noise=True):
        model = Sequential()
        if add_noise:
            model.add(GaussianNoise(0.05, input_shape=self.input_shape))
        # LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.3))
        
        # Add attention mechanism here
        model.add(MultiHeadAttention(num_heads=4, key_dim=50))
        model.add(LayerNormalization(epsilon=1e-6))  # Normalize the output of the attention layer
        
        # Final LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(units=1))  # Predicting 'hourly_calls'
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer='adam', loss=quantile_loss(0.7))
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])
        return history

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save_weights(self, filepath):
        """Save the model weights to the specified filepath."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load the model weights from the specified filepath."""
        self.model.load_weights(filepath)


# Class 4: ModelEvaluator
class ModelEvaluator:
    def __init__(self, scaler):
        self.scaler = scaler

    def inverse_transform(self, predictions, actual):
        # Create a placeholder array with zeros for the non-hourly_calls columns
        dummy_features = np.zeros((len(predictions), self.scaler.n_features_in_ - 1))
        
        # Concatenate predictions with dummy columns and inverse transform
        predicted_calls = self.scaler.inverse_transform(
            np.concatenate([predictions, dummy_features], axis=1))[:, 0]
        
        # Do the same for actual values
        actual_calls = self.scaler.inverse_transform(
            np.concatenate([actual.reshape(-1, 1), dummy_features], axis=1))[:, 0]
        
        return predicted_calls, actual_calls

    def plot_results(self, y_test_actual, predicted_calls):
        predicted_calls = np.expm1(predicted_calls)
        y_test_actual = np.expm1(y_test_actual)
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual Calls')
        plt.plot(predicted_calls, label='Predicted Calls')
        plt.title('Actual vs Predicted Number of Calls')
        plt.legend()
        plt.show()
        
    def plot_daily_results(self, data):
        # Group by day and plot each day separately
        data['date'] = data['timestamp'].dt.date
        for date, daily_data in data.groupby('date'):
            plt.figure(figsize=(10, 5))
            plt.plot(daily_data['timestamp'], np.expm1(daily_data['actual_calls']), label='Actual Calls')
            plt.plot(daily_data['timestamp'], np.expm1(daily_data['predicted_calls']), label='Predicted Calls')
            weekday_name = daily_data['timestamp'].dt.day_name().iloc[0]
            plt.title(f"Date: {date} ({weekday_name})")
            plt.xlabel("Hour")
            plt.ylabel("Calls")
            plt.legend()
            plt.show(block=True)  # Pause to view each plot

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
        plt.figure(figsize=(12, 12))
        sns.boxplot(x='month', y='daily_error', data=daily_totals)
        sns.stripplot(x='month', y='daily_error', data=daily_totals, color='red', jitter=0.2, size=2.5)
        
        # Calculate and plot the mean
        monthly_means = daily_totals.groupby('month')['daily_error'].mean().reset_index()
        sns.pointplot(x='month', y='daily_error', data=monthly_means, color='blue', markers='D', scale=1.0, linestyles='--')

        plt.title('Average Daily Error per Month')
        plt.xlabel('Month')
        plt.ylabel('Daily Error (%)')
        plt.xticks(rotation=45)
        plt.show()
