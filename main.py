####################################################
# main.py
#
# This script contains the main function for training and predicting
# the LSTM model. It also includes the argparse setup for command-line
# arguments to control the training and prediction processes.
####################################################

from src.data.data_loader import DataLoaderCustom
from src.models.lstm_model import LSTMModel, DataPreprocessorCustom, ModelEvaluatorCustom, QuantileLoss, TimeSeriesDataset
from src.predictor.predictor import Predictor
from src.utils.plot_functions import plot_avg_calls_per_weekday, plot_avg_calls_per_hour_by_weekday
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

def train_model(args):
    # Define file paths
    if not args.daily:
        file_path = 'data/callcenter_hourly.csv'
        train_file_path = 'data/train_data_hourly.csv'
        test_file_path = 'data/test_data_hourly.csv'
        weights_path = 'weights/lstm_weights_hourly.pth'
    else:
        file_path = 'data/callcenter_daily.csv'
        train_file_path = 'data/train_data_daily.csv'
        test_file_path = 'data/test_data_daily.csv'
        weights_path = 'weights/lstm_weights_daily.pth'
    
    # Load the full dataset
    data_loader = DataLoaderCustom(file_path)
    full_df = data_loader.load_data()

    if args.split:
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
        plot_avg_calls_per_weekday(full_df)
        plot_avg_calls_per_hour_by_weekday(full_df)
        return

    # Train the model if --train flag is set
    if args.train:
        # Load training data from split dataset
        data_loader = DataLoaderCustom(train_file_path)
        time_series_df = data_loader.load_data()

        # Preprocess Data
        data_preprocessor = DataPreprocessorCustom(hours=9, weekdays=5)
        time_series_df = data_preprocessor.add_lag_features(time_series_df)
        time_series_df = data_preprocessor.add_date_features(time_series_df)
        scaled_df = data_preprocessor.normalize_data(time_series_df)
        
        # Create sequences and split the data
        X, y = data_preprocessor.create_sequences(scaled_df.values)
        X_train, X_val, y_train, y_val = data_preprocessor.split_data(X, y)

        # Create PyTorch datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_dim=X_train.shape[2],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            add_noise=args.add_noise
        ).to(device)

        # Define loss function and optimizer with weight decay
        criterion = QuantileLoss(quantile=0.7)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Added weight_decay

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience = 10  # Number of epochs to wait for improvement
        epochs_no_improve = 0

        # Initialize lists for loss tracking
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            epoch_train_losses = []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_train_losses.append(loss.item())

            # Validation loss calculation
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device).unsqueeze(1)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)

            # Save losses for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Step scheduler
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), weights_path)
                print(f"Model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break

        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('losses.png')
        plt.show()

def main():
    # Step 1: Define argparse arguments
    parser = argparse.ArgumentParser(description='Process some options for PyTorch LSTM training and visualization.')

    # Add flags for training, prediction, splitting, and plotting
    parser.add_argument('--train', action='store_true', help='Train the LSTM model and save weights.')
    parser.add_argument('--predict', action='store_true', help='Generate predictions and plot results using saved weights.')

    # Add flags for model training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for model training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model training.')
    parser.add_argument('--hidden_dim', type=int, default=48, help='Hidden dimension size for LSTM.')  # Changed default to 48
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')  # Increased default from 0.3 to 0.5
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the data.')

    # Add flags for data processing
    parser.add_argument('--split', action='store_true', help='Split the dataset into train and test sets.')
    parser.add_argument('--plot_avg_calls', action='store_true', help='Plot the average calls per weekday.')

    # Add flag for daily data
    parser.add_argument('--daily', action='store_true', help='Use daily data for model training.')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.train:
        train_model(args)

    if args.predict:
        # Define file paths
        if not args.daily:
            test_file_path = 'data/test_data_hourly.csv'
            weights_path = 'weights/lstm_weights_hourly.pth'
        else:
            test_file_path = 'data/test_data_daily.csv'
            weights_path = 'weights/lstm_weights_daily.pth'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Predictor
        predictor = Predictor(
            model_weights_path=weights_path,
            data_file_path=test_file_path,
            input_dim=6,  # Number of features after preprocessing
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            add_noise=args.add_noise,
            quantile=0.7,
            device=device
        )
        predictor.predict_and_plot()

if __name__ == "__main__":
    main()