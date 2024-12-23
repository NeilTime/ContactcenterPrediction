import pandas as pd
import numpy as np

def DataPreprocessor(file_path, date_format):
    # Load the file, skipping the last 11 rows
    df = pd.read_excel(file_path, sheet_name=0, skipfooter=11)

    # Step 1: Find the starting point of the actual data ("Rijlabels")
    rijlabels_index = df[df.apply(lambda row: row.astype(str).str.contains('Rijlabels').any(), axis=1)].index[0]

    # Load the actual data from that point, again skipping the last 11 rows
    df = pd.read_excel(file_path, sheet_name=0, skiprows=rijlabels_index, skipfooter=11)

    # Step 2: Clean up the DataFrame
    # Assuming 'Rijlabels' is now the header and you have no other non-data rows
    df.columns = df.iloc[0]  # First row becomes the header
    df = df[1:]  # Remove the header row from the data
    df.reset_index(drop=True, inplace=True)

    # Step 3: Restructure the data for time-series processing
    data = []
    current_date = None

    # Loop through the DataFrame to reformat the data
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            # If 'Rijlabels' contains a date, this is a new day
            current_date = row['Rijlabels']  # Save the date
            # Parse the date with the provided date format
            current_date = pd.to_datetime(current_date, format=date_format)
            total_calls = row['Dialogue Volume']  # Get the total calls for the day
        else:
            # If 'Rijlabels' is NaN, this row represents hourly data
            hour = int(row['Rijlabels'])  # Extract the hour
            timestamp = f"{current_date.strftime('%d-%m-%Y')} {hour:02d}:00:00"  # Create a timestamp
            
            # Add the data to the new format
            data.append({
                'timestamp': pd.to_datetime(timestamp, format='%d-%m-%Y %H:%M:%S'),
                'total_calls': total_calls,
                'hourly_calls': row['Dialogue Volume'],
                'avg_talk_time': row['Average Initial Talktime'],
                'accepted_in_sla': row['Accepted in SLA'],
                'avg_queue_time': row['Average Queue Time']
            })

    # Convert the list to a DataFrame
    time_series_df = pd.DataFrame(data)

    # Step 4: Convert time columns ('avg_talk_time', 'avg_queue_time') to durations
    time_series_df['avg_talk_time'] = pd.to_timedelta(time_series_df['avg_talk_time'])
    time_series_df['avg_queue_time'] = pd.to_timedelta(time_series_df['avg_queue_time'])

    # Step 5: Extract the weekday (as an integer) from the timestamp
    time_series_df['weekday'] = time_series_df['timestamp'].dt.weekday

    # Step 6: Filter out weekends (Saturday=5, Sunday=6)
    time_series_df = time_series_df[time_series_df['weekday'] < 5]  # Keep only Monday to Friday

    return time_series_df

def CombineAndSaveData(file_path1, file_path2, csv_file_path):
    # Process each file with its respective date format
    df1 = DataPreprocessor(file_path1, "%d-%m-%Y")  # File with date format like "31-02-2002"
    df2 = DataPreprocessor(file_path2, "%d %b '%y")  # File with date format like "9 Sep '23"

    # Combine the data from both files
    combined_df = pd.concat([df1, df2])

    # Sort the combined DataFrame by the 'timestamp' column
    combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)

    # Save the combined and sorted data to CSV
    combined_df.to_csv(csv_file_path, index=False)

    return combined_df

if __name__ == "__main__":
    CombineAndSaveData(file_path2="/home/leon/project_haarlemmermeer/2023 dialogue volume per dag en uur.xlsx", file_path1="/home/leon/project_haarlemmermeer/2024 dialogue volume per dag en uur.xlsx", csv_file_path="/home/leon/project_haarlemmermeer/test.csv")