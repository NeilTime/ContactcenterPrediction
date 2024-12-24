import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

def plot_avg_calls_per_weekday(df):
    """
    Plots a bar plot of the average number of calls for each weekday, with error bars for standard deviation,
    and also plots individual total calls for each day to understand the spread of data.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    """
    # Group data by weekday and calculate the mean and standard deviation of hourly calls
    df = df[::9]  # Get only the first row of each day
    print(df)
    avg_calls_per_weekday = df.groupby('weekday')['total_calls'].mean()
    std_calls_per_weekday = df.groupby('weekday')['total_calls'].std()
    
    # Map weekday numbers to weekday names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Create a new column in df for weekday names (to simplify plotting individual points)
    df['weekday_name'] = df['weekday'].apply(lambda x: weekday_names[x])

    df['weekday_name'] = pd.Categorical(df['weekday_name'], categories=weekday_names, ordered=True)

    # Create the figure and axis objects
    plt.figure(figsize=(10, 6))

    # Plot the individual total calls for each day (scatter points)
    sns.stripplot(x='weekday_name', y='total_calls', data=df, jitter=True, color='gray', alpha=0.6, label='Total calls (each day)')

    # Plot the average with error bars (std)
    avg_calls_per_weekday.index = [weekday_names[i] for i in avg_calls_per_weekday.index]  # Map numbers to names
    plt.bar(avg_calls_per_weekday.index, avg_calls_per_weekday, yerr=std_calls_per_weekday, 
            capsize=5, color='skyblue', ecolor='black', alpha=0.7, label='Average calls (per weekday)')

    # Adding labels and title
    plt.title('Average Number of Calls per Weekday with Std Deviation and Individual Data Points', fontsize=16)
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Number of Daily Calls', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add a legend to distinguish individual points and averages
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_avg_calls_per_hour_by_weekday(df):
    """
    Plots a series of bar plots showing the average number of calls per hour for each weekday
    (one plot per weekday) with shared axes. Includes standard deviation as error bars and
    plots individual data points on top of the bars.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    """
    # Define the correct order of weekdays and map the weekday numbers to names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours_per_day = range(8, 17)
    
    # Create a new column for hours in the data (if not already available)
    df['hour'] = df['timestamp'].dt.hour
    
    # Create a figure with subplots, one for each weekday (5 subplots in total)
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(24, 6), sharey=True)

    for i, weekday in enumerate(weekday_names):
        # Filter the data for the current weekday
        weekday_data = df[df['weekday'] == i]
        
        # Group by hour and calculate the mean and std of hourly calls
        avg_calls_per_hour = weekday_data.groupby('hour')['hourly_calls'].mean()
        std_calls_per_hour = weekday_data.groupby('hour')['hourly_calls'].std()
        
        # Plot a barplot for each weekday's average hourly calls with error bars (standard deviation)
        axs[i].bar(avg_calls_per_hour.index, avg_calls_per_hour.values, 
                   yerr=std_calls_per_hour.values, capsize=4, color='skyblue', alpha=0.6)
        
        # Overlay individual data points (scatter plot) on top of the bars
        axs[i].scatter(weekday_data['hour'], weekday_data['hourly_calls'], 
                       color='darkblue', alpha=0.8, s=10, label='Individual Data')

        # Set the title and labels for the subplot
        axs[i].set_title(weekday)
        axs[i].set_xlabel('Hour of the Day')
        axs[i].set_xticks(hours_per_day)
        
    # Set the shared y-axis label
    axs[0].set_ylabel('Average Number of Hourly Calls')

    # Add a legend for the individual data points
    axs[0].legend(loc='upper right')

    print(df)
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test_actual, predicted_calls):
    plt.figure(figsize=(14, 7))
    
    # Create a long-form DataFrame for plotting
    df_plot = pd.DataFrame({
        'Type': ['Actual Calls'] * len(y_test_actual) + ['Predicted Calls'] * len(predicted_calls),
        'Calls': np.concatenate([y_test_actual, predicted_calls])
    })
    
    # Box Plot
    sns.boxplot(x='Type', y='Calls', data=df_plot, showmeans=True, meanline=True, 
                meanprops={"linewidth": 2, "color": "red", "linestyle": "--"})
    
    # Individual Data Points
    sns.stripplot(x='Type', y='Calls', data=df_plot, color='black', alpha=0.5, jitter=True, dodge=True)
    
    # Mean Lines
    means = df_plot.groupby('Type')['Calls'].mean().values
    plt.plot([0, 1], means, color='red', marker='o', linestyle='--', linewidth=2, markersize=8, label='Mean')
    
    # Adding Titles and Labels
    plt.title('Actual vs Predicted Number of Calls', fontsize=16)
    plt.ylabel('Number of Calls', fontsize=14)
    plt.xlabel('')

    # Creating Custom Legend Handles
    box_patch = mpatches.Patch(color='C0', label='Box Plot')  # 'C0' is default first color
    strip = mpatches.Patch(color='black', label='Individual Data Points')
    mean_line = mpatches.Patch(color='red', label='Mean')
    plt.legend(handles=[box_patch, strip, mean_line], title='Legend', loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_error_distributions(daily_totals, monthly_means):
    """
    Plots error distributions per month using box plots and overlays individual data points.

    Args:
        daily_totals (pandas.DataFrame): DataFrame containing daily error metrics.
        monthly_means (pandas.DataFrame): DataFrame containing monthly mean errors.
    """
    if monthly_means is None or daily_totals is None:
        raise ValueError("Both 'daily_totals' and 'monthly_means' DataFrames must be provided.")
    
    # Convert 'month' period to string for plotting
    monthly_means['month'] = monthly_means['month'].astype(str)
    daily_totals['month'] = daily_totals['month'].astype(str)
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='month', y='daily_error', data=daily_totals, palette="Set3")
    sns.pointplot(x='month', y='daily_error', data=monthly_means, color='red', markers='D', 
                 linestyles='-', join=True, label='Monthly Mean')
    
    plt.title('Daily Prediction Error Distribution per Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Daily Error (%)', fontsize=14)
    
    # Rotate X-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Creating Custom Legend Handles
    box_patch = mpatches.Patch(color='C0', label='Daily Error Distribution')  # Default color
    strip = mpatches.Patch(color='black', label='Individual Days')
    mean_line = mpatches.Patch(color='red', label='Monthly Mean')
    plt.legend(handles=[box_patch, strip, mean_line], title='Legend', loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_highest_and_lowest_error_days(data):
    """
    Plots the actual vs predicted calls for the days with the highest and lowest daily errors.

    Args:
        data (pandas.DataFrame): DataFrame containing 'timestamp', 'actual_calls',
                                  'predicted_calls', 'daily_error', and 'date' columns.
    """
    required_columns = {'timestamp', 'actual_calls', 'predicted_calls', 'daily_error', 'date'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"DataFrame must contain 'timestamp', 'actual_calls', 'predicted_calls', 'daily_error', and 'date' columns.")
    
    # Identify the day with the highest daily error
    highest_error_day = data.loc[data['daily_error'].idxmax()]
    highest_error_date = highest_error_day['date']
    highest_error_value = highest_error_day['daily_error']
    
    # Identify the day with the lowest daily error
    lowest_error_day = data.loc[data['daily_error'].idxmin()]
    lowest_error_date = lowest_error_day['date']
    lowest_error_value = lowest_error_day['daily_error']
    
    # Filter the data for the highest error day
    data_high = data[data['date'] == highest_error_date]
    
    # Filter the data for the lowest error day
    data_low = data[data['date'] == lowest_error_date]
    
    # Debugging statements
    print(f"Highest error date: {highest_error_date}, Error: {highest_error_value:.2f}%")
    print(f"Lowest error date: {lowest_error_date}, Error: {lowest_error_value:.2f}%")
    print(f"Data for highest error day:\n{data_high}")
    print(f"Data for lowest error day:\n{data_low}")
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot for Highest Error Day
    axes[0].plot(data_high['timestamp'].dt.hour, data_high['actual_calls'], marker='o', label='Actual Calls')
    axes[0].plot(data_high['timestamp'].dt.hour, data_high['predicted_calls'], marker='s', label='Predicted Calls')
    axes[0].set_title(f'Actual vs Predicted Calls on Highest Error Day: {highest_error_date} ({highest_error_value:.2f}%)', fontsize=14)
    axes[0].set_xlabel('Working Hour', fontsize=12)
    axes[0].set_ylabel('Number of Calls', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot for Lowest Error Day
    axes[1].plot(data_low['timestamp'].dt.hour, data_low['actual_calls'], marker='o', label='Actual Calls')
    axes[1].plot(data_low['timestamp'].dt.hour, data_low['predicted_calls'], marker='s', label='Predicted Calls')
    axes[1].set_title(f'Actual vs Predicted Calls on Lowest Error Day: {lowest_error_date} ({lowest_error_value:.2f}%)', fontsize=14)
    axes[1].set_xlabel('Working Hour', fontsize=12)
    axes[1].set_ylabel('Number of Calls', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_monthly_predicted_vs_actual(data):
    """
    Plots subplots for each month showing predicted vs actual calls across hours.
    Each subplot includes:
        - Mean of actual calls as a blue line.
        - Mean of predicted calls as an orange line.
        - Shaded areas representing ±1 standard deviation for both actual and predicted calls.
    
    Args:
        data (pandas.DataFrame): DataFrame containing 'timestamp', 'actual_calls', and 'predicted_calls' columns.
    """
    # Ensure 'timestamp' is datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Extract month and hour
    data['month'] = data['timestamp'].dt.month
    data['hour'] = data['timestamp'].dt.hour
    
    # Define month names for better readability
    month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                  5: "May", 6: "June", 7: "July", 8: "August",
                  9: "September", 10: "October", 11: "November", 12: "December"}
    data['month_name'] = data['month'].map(month_names)
    
    # Get list of unique months in the data
    unique_months = sorted(data['month'].unique())
    
    # Determine the number of rows and columns for subplots
    num_months = len(unique_months)
    num_cols = 3
    num_rows = (num_months + num_cols - 1) // num_cols  # Ceiling division
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration
    
    for idx, month in enumerate(unique_months):
        ax = axes[idx]
        month_data = data[data['month'] == month]
        
        # Group by hour and calculate mean and std
        grouped = month_data.groupby('hour').agg(
            actual_mean=('actual_calls', 'mean'),
            actual_std=('actual_calls', 'std'),
            predicted_mean=('predicted_calls', 'mean'),
            predicted_std=('predicted_calls', 'std')
        ).reset_index()
        
        # Plot actual calls
        ax.plot(grouped['hour'], grouped['actual_mean'], label='Actual Calls', color='blue')
        ax.fill_between(grouped['hour'],
                        grouped['actual_mean'] - grouped['actual_std'],
                        grouped['actual_mean'] + grouped['actual_std'],
                        color='blue', alpha=0.2)
        
        # Plot predicted calls
        ax.plot(grouped['hour'], grouped['predicted_mean'], label='Predicted Calls', color='orange')
        ax.fill_between(grouped['hour'],
                        grouped['predicted_mean'] - grouped['predicted_std'],
                        grouped['predicted_mean'] + grouped['predicted_std'],
                        color='orange', alpha=0.2)
        
        # Set title and labels
        ax.set_title(f"{month_names[month]}")
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Calls')
        ax.legend()
        ax.grid(True)
    
    # Remove any empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Predicted vs Actual Calls per Hour for Each Month', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.show()