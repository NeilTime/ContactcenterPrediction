####################################################
# predictor.py
#
# This script contains the Predictor class that 
# loads the LSTM model weights and calculates
# plots and accuracies.
####################################################

import numpy as np
import pandas as pd
from lstm_model import DataLoader, DataPreprocessor, LSTMModel, ModelEvaluator
import matplotlib.pyplot as plt
import datetime

class Predictor:
    def __init__(self, model_weights_path, data_file_path):
        self.model_weights_path = model_weights_path
        self.data_file_path = data_file_path  # Now points to test_data.csv for testing
        self.data_loader = DataLoader(data_file_path)
        self.data_preprocessor = DataPreprocessor(hours=9, weekdays=5)
    
    def load_data_and_prepare_sequences(self):
        # Load and preprocess data from the test set
        time_series_df = self.data_loader.load_data()
        time_series_df = self.data_preprocessor.add_lag_features(time_series_df)
        time_series_df = self.data_preprocessor.add_date_features(time_series_df)
        scaled_df = self.data_preprocessor.normalize_data(time_series_df)
        
        # Create sequences without splitting (only test data is used here)
        X, y = self.data_preprocessor.create_sequences(scaled_df.values)
        return X, y, time_series_df.iloc[-len(y):]  # Return corresponding dates as well

    
    def load_model_and_predict(self, X_test):
        # Initialize the LSTM model and load weights
        input_shape = (X_test.shape[1], X_test.shape[2])
        lstm_model = LSTMModel(input_shape=input_shape, add_noise=True)
        lstm_model.load_weights(self.model_weights_path)
        
        # Make predictions
        predictions = lstm_model.predict(X_test)
        return predictions
    
    def evaluate_and_plot(self, predictions, y_test, dates):
        # Initialize ModelEvaluator for inverse transforming and plotting
        model_evaluator = ModelEvaluator(scaler=self.data_preprocessor.scaler)
        predicted_calls, y_test_actual = model_evaluator.inverse_transform(predictions, y_test)
        
        # Separate the data by each day for plotting
        dates['predicted_calls'] = np.expm1(predicted_calls)
        dates['actual_calls'] = np.expm1(y_test_actual)
        
        # Calculate error metrics and plot daily results
        # model_evaluator.plot_daily_results(dates)
        model_evaluator.calculate_average_error(dates)
        model_evaluator.plot_results(y_test_actual=y_test_actual, predicted_calls=predicted_calls)

    def predict_and_plot(self):
        # Load data, prepare sequences, make predictions, and plot results
        X_test, y_test, dates = self.load_data_and_prepare_sequences()
        predictions = self.load_model_and_predict(X_test)
        self.evaluate_and_plot(predictions, y_test, dates)
