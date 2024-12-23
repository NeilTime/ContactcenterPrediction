import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_avg_calls_per_weekday(df):
    """
    Plots a bar plot of the average number of calls for each weekday, with error bars for standard deviation,
    and also plots individual total calls for each day to understand the spread of data.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    """
    # Group data by weekday and calculate the mean and standard deviation of hourly calls
    df = df[::9]  # Get only the first row of each day
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
    
    # Create a figure with subplots, one for each weekday (7 subplots in total)
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
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
