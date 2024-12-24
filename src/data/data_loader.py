import pandas as pd

class DataLoaderCustom:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load the data from the specified file path
        time_series_df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        time_series_df.sort_values(by='timestamp', inplace=True)
        return time_series_df