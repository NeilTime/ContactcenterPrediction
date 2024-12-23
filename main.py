from lstm_model import DataLoader, DataPreprocessor, LSTMModel
from predictor import Predictor
from plot_functions import plot_avg_calls_per_weekday, plot_avg_calls_per_hour_by_weekday
import pandas as pd
import argparse

def main():
    # Step 1: Define argparse arguments
    parser = argparse.ArgumentParser(description='Process some options for LSTM training and visualization.')

    # Add flags for training, prediction, splitting, and plotting
    parser.add_argument('--train', action='store_true', help='Train the LSTM model and save weights.')
    parser.add_argument('--predict', action='store_true', help='Generate predictions and plot results using saved weights.')
    parser.add_argument('--split', action='store_true', help='Split the dataset into train and test sets.')
    parser.add_argument('--plot_avg_calls', action='store_true', help='Plot the average calls per weekday.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for model training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model training.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define file paths
    file_path = 'call_center_data.csv'
    train_file_path = 'train_data.csv'
    test_file_path = 'test_data.csv'
    weights_path = 'lstm_weights.weights.h5'

    # Split the data if --split flag is set
    if args.split:
        # Load the full dataset
        data_loader = DataLoader(file_path)
        full_df = data_loader.load_data()

        # Perform the split, with 80% for training and 20% for testing
        train_size = int(len(full_df) * 0.8)
        train_df = full_df.iloc[:train_size]
        test_df = full_df.iloc[train_size:]

        # Save to CSV
        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)
        print(f"Data split into {train_file_path} and {test_file_path}.")
        return  # Exit after splitting

    # Plot average calls if requested
    if args.plot_avg_calls:
        data_loader = DataLoader(file_path)
        time_series_df = data_loader.load_data()
        plot_avg_calls_per_weekday(time_series_df)
        plot_avg_calls_per_hour_by_weekday(time_series_df)
        return

    # Train the model if --train flag is set
    if args.train:
        # Load training data from split dataset
        data_loader = DataLoader(train_file_path)
        time_series_df = data_loader.load_data()

        # Preprocess Data
        data_preprocessor = DataPreprocessor(hours=9, weekdays=5)
        time_series_df = data_preprocessor.add_lag_features(time_series_df)
        time_series_df = data_preprocessor.add_date_features(time_series_df)
        scaled_df = data_preprocessor.normalize_data(time_series_df)
        
        # Create sequences and split the data
        X, y = data_preprocessor.create_sequences(scaled_df.values)
        X_train, X_test, y_train, y_test = data_preprocessor.split_data(X, y)

        # Build and Train the LSTM Model
        lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]), add_noise=True)
        lstm_model.train(X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)

        # Save the trained model weights
        lstm_model.save_weights(weights_path)

    # Predict using saved weights if --predict flag is set
    if args.predict:
        predictor = Predictor(model_weights_path=weights_path, data_file_path=test_file_path)
        predictor.predict_and_plot()

if __name__ == "__main__":
    main()