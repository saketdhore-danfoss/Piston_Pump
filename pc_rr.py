import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, savgol_filter, argrelextrema
import pandas as pd
import os
from nptdms import TdmsFile
import mplcursors
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import re
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq
import pywt
from scipy.signal import butter, filtfilt
from PyEMD import EMD

def read_tdms(filepath):
    """
    Reads a TDMS file and converts it into a cleaned pandas DataFrame.

    Parameters:
    filepath (str): The path to the TDMS file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the TDMS file, with columns cleaned and NaN-only columns removed.
    """
    # Read the TDMS file
    tdms_file = TdmsFile.read(filepath)

    # Convert the TDMS file to a pandas DataFrame
    df = tdms_file.as_dataframe()

    # Remove columns that only contain NaN values
    df = df.dropna(axis=1, how='all')

    # Clean up column names by removing unwanted characters and trimming whitespace
    df.columns = df.columns.str.replace("/", "")
    df.columns = df.columns.str.replace("'", "")
    df.columns = df.columns.str.replace("Data", "")
    df.columns = df.columns.str.strip()

    return df





def plot_segments(df, t, s, p, column_name, n=7, min_distance=2000, segment_padding=4000):
    """
    Plots segments around the highest and lowest values in a specified column of a DataFrame,
    removing outliers that are more than a specified number of standard deviations away from the mean.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    t : temperature
    s : speed
    pressure : p
    n (int): The number of highest and lowest values to find. Always set it to be the upper bound of no. cycles
    min_distance (int): The minimum distance between the indices of the extreme values.
    segment_padding (int): The number of indices to include before and after the extreme values in the segments.

    Returns:
    list: A list of tuples containing the start and end indices of the segments.
    """
    y = df[column_name].values

    def find_extreme_values(y, n, min_distance, std_dev_threshold, find_highest=True):
        """
        Finds the n highest/lowest values separated by at least min_distance indices.

        Parameters:
        y (np.array): The array of values.
        n (int): The number of extreme values to find.
        min_distance (int): The minimum distance between the indices of the extreme values.
        std_dev_threshold (float): The number of standard deviations to use for outlier detection.
        find_highest (bool): Whether to find the highest values (True) or the lowest values (False).

        Returns:
        tuple: A tuple containing the filtered extreme values and their indices.
        """
        extreme_values = []
        extreme_indices = []

        # Sort the values and their indices
        sorted_indices = np.argsort(y)[::-1] if find_highest else np.argsort(y)

        for idx in sorted_indices:
            if len(extreme_indices) == 0 or all(abs(idx - i) >= min_distance for i in extreme_indices):
                extreme_values.append(y[idx])
                extreme_indices.append(idx)
            if len(extreme_indices) == n:
                break

        # Calculate mean and standard deviation
        mean_val = np.mean(extreme_values)
        std_val = np.std(extreme_values)

        # Remove outliers that are more than std_dev_threshold standard deviations away from the mean
        filtered_values = [val for val in extreme_values if abs(val - mean_val) <= std_dev_threshold * std_val]
        filtered_indices = [extreme_indices[i] for i, val in enumerate(extreme_values) if
                            abs(val - mean_val) <= std_dev_threshold * std_val]

        return filtered_values, filtered_indices

    # Try different standard deviation thresholds
    std_dev_thresholds = [1, 1.5, 2]
    for std_dev_threshold in std_dev_thresholds:
        # Find the n highest values separated by at least min_distance indices
        highest_values, highest_indices = find_extreme_values(y, n, min_distance, std_dev_threshold, find_highest=True)

        # Find the n lowest values separated by at least min_distance indices
        lowest_values, lowest_indices = find_extreme_values(y, n, min_distance, std_dev_threshold, find_highest=False)

        # Check if the lengths of highest and lowest values match
        if len(highest_indices) == len(lowest_indices):
            break
    else:
        raise ValueError("Values are too erroneous. Unable to find matching highest and lowest values.")

    # Ensure the highest and lowest indices are sorted
    highest_indices.sort()
    lowest_indices.sort()

    # Isolate the pairs and extract the required segments
    segments = []
    for peak_idx, trough_idx in zip(highest_indices, lowest_indices):
        start_idx = max(0, peak_idx - segment_padding)
        end_idx = min(len(y), trough_idx + segment_padding)
        segments.append((start_idx, end_idx))

    # Create subplots
    num_segments = len(segments)
    num_rows = (num_segments + 1) // 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Plot each segment in a separate subplot
    for i, (start_idx, end_idx) in enumerate(segments):
        axes[i].plot(range(start_idx, end_idx), y[start_idx:end_idx], 'b-', linewidth=2)
        axes[i].set_title(f'Segment {i + 1}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel(column_name)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f"Plot for Temp ({t}), Speed ({s}), Pressure ({p})")

    # Generate the filename based on TSP
    filename = f"segmentation_T{t}_S{s}_P{p}.png"

    # Save the plot with the generated filename
    plt.savefig(filename, format="png", dpi=400)
    print(f"Plot saved as {filename}")

    return segments


def plot_pressure_analysis(segment_df, target_pressure, threshold_percentage, deadhead_pressure, ax, t, s, p):
    """
    Analyzes and plots the outlet pressure data over time, highlighting key points such as max/min pressures,
    intersection points, and recovery pressures.

    Parameters:
    segment_df (pandas.DataFrame): DataFrame containing the time and pressure data.
    target_pressure (float): The target pressure value. (generally 75% of deadhead pressure)
    threshold_percentage (float): The percentage threshold used to determine the target pressure range.
    deadhead_pressure (float): The deadhead pressure value.
    ax (matplotlib.axes.Axes): The matplotlib axes object where the plot will be drawn.
    t, s, p: temperature, speed and pressure.

    Returns:
    dict: A dictionary containing key points of interest:
        - 'max_point': Tuple of (time, pressure) for the maximum pressure point.
        - 'min_point': Tuple of (time, pressure) for the minimum pressure point.
        - 'intersection_points': List of tuples (time, pressure) for selected intersection points.
        - 'gradient_mean_point': Tuple of (time, pressure) for the recovery pressure point.
        - 'target_point': Tuple of (time, pressure) for the target pressure point.
        - 'deadhead_pressure_mean': Mean value of the deadhead pressure during specific conditions.
    matplotlib.axes.Axes: The axes object with the plot.
    """
    # Initialize variables
    intersection_times = []
    intersection_pressures = []

    # Extract the time and pressure data
    time = segment_df['Time_ms']
    pressure = segment_df['Outlet_Pressure_Psi']

    # Use the raw pressure data without smoothing
    smoothed_pressure = pressure

    # Plot the main line
    ax.plot(time, smoothed_pressure, label='Outlet Pressure')

    # Find the max and min points
    max_index = smoothed_pressure.argmax()
    min_index = smoothed_pressure.argmin()

    max_time = time[max_index]
    max_pressure = smoothed_pressure[max_index]

    min_time = time[min_index]
    min_pressure = smoothed_pressure[min_index]

    # Plot the max and min points
    ax.scatter(max_time, max_pressure, color='red', label='Max Pressure')
    ax.scatter(min_time, min_pressure, color='blue', label='Min Pressure')

    # Calculate the gradient
    grads = np.gradient(smoothed_pressure)

    # Find specific indices where the gradient is between -0.1 and 0.1 and pressure is within 3% of deadhead_pressure
    specific_indices = []
    for i, grad in enumerate(grads):
        if -0.1 <= grad <= 0.1 and deadhead_pressure * 0.97 <= smoothed_pressure[i] <= deadhead_pressure * 1.03:
            specific_indices.append(i)

    # Extract the times and pressures for these specific indices
    specific_times = time[specific_indices]
    specific_pressures = smoothed_pressure[specific_indices]
    deadhead_pressure_mean = specific_pressures.mean()

    # Fit a line through these points
    if len(specific_times) > 1:  # Ensure there are enough points to fit a line
        specific_times_reshaped = np.array(specific_times).reshape(-1, 1)
        model = LinearRegression().fit(specific_times_reshaped, specific_pressures)

        # Extend the line
        extended_times = np.linspace(time.min(), time.max(), 1000).reshape(-1, 1)
        extended_line = model.predict(extended_times)

        # Interpolate the extended line to match the original time indices
        interpolated_line = np.interp(time, extended_times.flatten(), extended_line)

        # Plot the extended line
        ax.plot(time, interpolated_line, color='orange', label='Extended Fitted Line')

        # Find intersection points
        intersection_indices = []
        for i in range(1, len(time)):
            if (smoothed_pressure[i - 1] < interpolated_line[i - 1] and smoothed_pressure[i] > interpolated_line[i]) or \
                    (smoothed_pressure[i - 1] > interpolated_line[i - 1] and smoothed_pressure[i] < interpolated_line[i]):
                if len(intersection_indices) == 0 or (time[i] - time[intersection_indices[-1]]) >= 0:
                    intersection_indices.append(i)

        # Select only the first two and the last intersection points
        if len(intersection_indices) > 2:
            selected_indices = intersection_indices[:2] + [intersection_indices[-1]]
        else:
            selected_indices = intersection_indices

        intersection_times = time[selected_indices]
        intersection_pressures = smoothed_pressure[selected_indices]

        # Plot the selected intersection points
        ax.scatter(intersection_times, intersection_pressures, color='purple', label='Selected Intersection Points')

    # Calculate the threshold value
    threshold = threshold_percentage * target_pressure

    # Find the first point after the min point where the pressure is just greater than or equal to target_pressure - threshold
    target_index = None
    for i in range(min_index + 1, len(smoothed_pressure)):
        if smoothed_pressure[i] >= (target_pressure - threshold):
            target_index = i
            break

    if target_index is not None:
        target_time = time[target_index]
        target_pressure_value = smoothed_pressure[target_index]

    # Find points where the gradient is between -0.1 and 0.1 after the min point
    gradient_indices = []
    for i in range(min_index + 1, len(grads)):
        if -0.1 <= grads[i] <= 0.1:
            gradient_indices.append(i)

    gradient_times = time[gradient_indices]
    gradient_pressures = smoothed_pressure[gradient_indices]
    gradient_pressures_mean = gradient_pressures.mean()

    # Plot the gradient points
    # ax.scatter(gradient_times, gradient_pressures, color='green', label='Zero slope recovery')

    # Add labels and legend
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Outlet Pressure (Psi)')
    ax.set_title('Outlet Pressure Analysis')
    ax.legend()

    # Find the first point after the min point where the pressure is just greater than or equal to the mean gradient pressure
    gradient_mean_index = None
    for i in range(min_index + 1, len(smoothed_pressure)):
        if smoothed_pressure[i] >= gradient_pressures_mean:
            gradient_mean_index = i
            break

    if gradient_mean_index is not None:
        gradient_mean_time = time[gradient_mean_index]
        gradient_mean_pressure = smoothed_pressure[gradient_mean_index]
        ax.scatter(gradient_mean_time, gradient_mean_pressure, color='magenta', label='Recovery Pressure')

    # Return all points of interest and the axes object
    points_of_interest = {
        'max_point': (max_time, max_pressure),
        'min_point': (min_time, min_pressure),
        'intersection_points': list(zip(intersection_times, intersection_pressures)),
        'gradient_mean_point': (gradient_mean_time, gradient_mean_pressure) if gradient_mean_index is not None else None,
        'target_point': (target_time, target_pressure_value) if target_index is not None else None,
        'deadhead_pressure_mean': deadhead_pressure_mean
    }

    return points_of_interest, ax

def extract_speed_pressure_temperature(filepath):
    # Define regular expression patterns to extract temperature, speed, and pressure
    temp_pattern = re.compile(r'(\d+)\s*F', re.IGNORECASE)
    speed_pattern = re.compile(r'(\d+)\s*rpm', re.IGNORECASE)
    pressure_pattern = re.compile(r'(\d+)\s*psi', re.IGNORECASE)

    # Search for the patterns in the filepath
    temp_match = temp_pattern.search(filepath)
    speed_match = speed_pattern.search(filepath)
    pressure_match = pressure_pattern.search(filepath)

    # Extract the matched groups
    temperature = int(temp_match.group(1)) if temp_match else None
    speed = int(speed_match.group(1)) if speed_match else None
    pressure = int(pressure_match.group(1)) if pressure_match else None

    return temperature, speed, pressure


def get_threshold_percentage(pressure):
    """
    Returns the threshold percentage based on the given pressure value.

    Parameters:
    pressure (float): The pressure value.

    Returns:
    float: The threshold percentage.
    """
    if pressure == 5076:
        return 0.01
    elif pressure == 4060:
        return 0.03
    elif pressure == 3045:
        return 0.05
    else:
        return 0.01

def main():
    directory_mode = 1

    if directory_mode == 1:
        results_df = pd.DataFrame()
        for root,dirs,files in os.walk(r"C:\Piston_Pump_Efficiency\pc_rr_data"):
            for file in files:
                if file.endswith(".tdms"):

                    print("Processing file: "+file)
                    df = read_tdms(os.path.join(root,file))
                    #statistics_df = pd.DataFrame()
                    temp,speed,pressure = extract_speed_pressure_temperature(file)
                    target_pressure = pressure * 0.75
                    threshold_percent = get_threshold_percentage(pressure)
                    deadhead_pressure = pressure
                    segments = plot_segments(df, temp, speed, pressure, 'Outlet_Pressure_Psi', 7)

                    # Number of segments
                    num_segments = len(segments)

                    # Calculate the number of rows and columns for subplots
                    num_cols = 2
                    num_rows = (num_segments + num_cols - 1) // num_cols  # This ensures enough rows to fit all segments

                    # Create subplots
                    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
                    axes = axes.flatten()

                    points_list = []
                    plots_list = []

                    for i in range(num_segments):
                        sliced_df = df.iloc[segments[i][0]:segments[i][1]].reset_index(drop=True)
                        points, plot = plot_pressure_analysis(sliced_df, target_pressure, threshold_percent, deadhead_pressure, axes[i],temp,speed,pressure)
                        points_list.append(points)
                        plots_list.append(plot)

                    # Hide any unused subplots
                    for j in range(num_segments, len(axes)):
                        fig.delaxes(axes[j])

                    # Adjust layout
                    plt.tight_layout()
                    plt.title(f"Plot for Temp ({temp}), Speed ({speed}), Pressure ({pressure})")
                    filename = f"pc_rr_T{temp}_S{speed}_P{pressure}.png"

                    # Save the plot with the generated filename
                    plt.savefig(filename, format="png", dpi=400)
                    print(f"Plot saved as {filename}")

                    # Print points of interest for each segment
                    for i, points in enumerate(points_list):
                        #print(f"Segment {i + 1}:")
                        #print(points)
                        max_pressure_t= points['max_point'][0]
                        max_pressure_v = points['max_point'][1]
                        min_pressure_t= points['min_point'][0]
                        min_pressure_v = points['min_point'][1]

                        start_response_time_t = points['intersection_points'][0][0]
                        start_response_time_v = points['intersection_points'][0][1]
                        end_response_time_t = points['intersection_points'][1][0]
                        end_response_time_v = points['intersection_points'][1][1]
                        start_recovery_time_t = points['intersection_points'][2][0]
                        start_recovery_time_v = points['intersection_points'][2][1]
                        end_recovery_time_t = points['gradient_mean_point'][0]
                        end_recovery_time_v = points['gradient_mean_point'][1]

                        deadhead_pressure_mean = points['deadhead_pressure_mean']
                        new_row = {
                            "File": file,
                            "Temp": temp,
                            "Speed": speed,
                            "Pressure": pressure,
                            "Cycle No.": i + 1,
                            "Time @ Max Pressure": max_pressure_t,
                            "Pressure @ Max Pressure" : max_pressure_v,
                            "Time @ Min Pressure": min_pressure_t,
                            "Pressure @ Min Pressure" : min_pressure_v,
                            "Time @ Start of Response Time":start_response_time_t,
                            "Pressure @ Start of Response Time": start_response_time_v,
                            "Time @ End of Response Time": end_response_time_t,
                            "Pressure @ End of Response Time": end_response_time_v,
                            "Time @ Start of Recovery Time": start_recovery_time_t,
                            "Pressure @ Start of Recovery Time": start_recovery_time_v,
                            "Time @ End of Recovery Time":end_recovery_time_t,
                            "Pressure @ End of Recovery Time": end_recovery_time_v,
                            "Mean deahead Pressure":deadhead_pressure_mean,
                            "Response Time":end_response_time_t - start_response_time_t,
                            "Recovery Time": end_recovery_time_t-start_recovery_time_t
                        }

                        # Append the new row to the DataFrame
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_csv(r"all_results.csv")
        grouped_df = results_df.groupby(['File', 'Temp', 'Speed','Pressure']).apply(lambda x: x).reset_index(drop=True)

        # Write the grouped DataFrame to an Excel file
        output_file = 'beautified_results.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            grouped_df.to_excel(writer, index=False, sheet_name='Results')

            # Optional: Beautify the Excel file
            workbook = writer.book
            worksheet = writer.sheets['Results']

            # Set the column width for better readability
            for col_num, value in enumerate(grouped_df.columns.values):
                max_len = max(grouped_df[value].astype(str).map(len).max(), len(value)) + 2
                worksheet.set_column(col_num, col_num, max_len)

            # Add any additional formatting if needed
            # Example: Set header format
            header_format = workbook.add_format(
                {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(grouped_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

        print(f"Beautified Excel file saved as {output_file}")
    else:
        filename = r'C:\Piston_Pump_Efficiency\pc_rr_data\your_file.tdms'
        # Check if the file is a .tdms file
        if filename.endswith(".tdms"):
            print("Processing file: " + filename)

            # Initialize an empty DataFrame to store results
            results_df = pd.DataFrame()

            # Read the TDMS file
            df = read_tdms(filename)

            # Extract temperature, speed, and pressure from the filename
            temp, speed, pressure = extract_speed_pressure_temperature(filename)

            # Define target pressure and threshold percentage
            target_pressure = pressure * 0.75
            threshold_percent = get_threshold_percentage(pressure)
            deadhead_pressure = pressure

            # Plot extreme segments
            segments = plot_segments(df, temp, speed, pressure, 'Outlet_Pressure_Psi', 7)

            # Number of segments
            num_segments = len(segments)

            # Calculate the number of rows and columns for subplots
            num_cols = 2
            num_rows = (num_segments + num_cols - 1) // num_cols  # This ensures enough rows to fit all segments

            # Create subplots
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten()

            points_list = []
            plots_list = []

            for i in range(num_segments):
                sliced_df = df.iloc[segments[i][0]:segments[i][1]].reset_index(drop=True)
                points, plot = plot_pressure_analysis(sliced_df, target_pressure, threshold_percent, deadhead_pressure,
                                                      axes[i], temp, speed, pressure)
                points_list.append(points)
                plots_list.append(plot)

            # Hide any unused subplots
            for j in range(num_segments, len(axes)):
                fig.delaxes(axes[j])

            # Adjust layout
            plt.tight_layout()
            plt.title(f"Plot for Temp ({temp}), Speed ({speed}), Pressure ({pressure})")
            plot_filename = f"pc_rr_T{temp}_S{speed}_P{pressure}.png"

            # Save the plot with the generated filename
            plt.savefig(plot_filename, format="png", dpi=400)
            print(f"Plot saved as {plot_filename}")

            # Print points of interest for each segment
            for i, points in enumerate(points_list):
                # print(f"Segment {i + 1}:")
                # print(points)
                max_pressure_t = points['max_point'][0]
                max_pressure_v = points['max_point'][1]
                min_pressure_t = points['min_point'][0]
                min_pressure_v = points['min_point'][1]

                start_response_time_t = points['intersection_points'][0][0]
                start_response_time_v = points['intersection_points'][0][1]
                end_response_time_t = points['intersection_points'][1][0]
                end_response_time_v = points['intersection_points'][1][1]
                start_recovery_time_t = points['intersection_points'][2][0]
                start_recovery_time_v = points['intersection_points'][2][1]
                end_recovery_time_t = points['gradient_mean_point'][0]
                end_recovery_time_v = points['gradient_mean_point'][1]

                deadhead_pressure_mean = points['deadhead_pressure_mean']
                new_row = {
                    "File": filename,
                    "Temp": temp,
                    "Speed": speed,
                    "Pressure": pressure,
                    "Cycle No.": i + 1,
                    "Time @ Max Pressure": max_pressure_t,
                    "Pressure @ Max Pressure": max_pressure_v,
                    "Time @ Min Pressure": min_pressure_t,
                    "Pressure @ Min Pressure": min_pressure_v,
                    "Time @ Start of Response Time": start_response_time_t,
                    "Pressure @ Start of Response Time": start_response_time_v,
                    "Time @ End of Response Time": end_response_time_t,
                    "Pressure @ End of Response Time": end_response_time_v,
                    "Time @ Start of Recovery Time": start_recovery_time_t,
                    "Pressure @ Start of Recovery Time": start_recovery_time_v,
                    "Time @ End of Recovery Time": end_recovery_time_t,
                    "Pressure @ End of Recovery Time": end_recovery_time_v,
                    "Deadhead Mean":deadhead_pressure_mean,
                    "Response Time": end_response_time_t - start_response_time_t,
                    "Recovery Time": end_recovery_time_t - start_recovery_time_t
                }

                # Append the new row to the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(r"all_results.csv")
            grouped_df = results_df.groupby(['File', 'Temp', 'Speed', 'Pressure']).apply(lambda x: x).reset_index(
                drop=True)

            # Write the grouped DataFrame to an Excel file
            output_file = 'beautified_results.xlsx'
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                grouped_df.to_excel(writer, index=False, sheet_name='Results')

                # Optional: Beautify the Excel file
                workbook = writer.book
                worksheet = writer.sheets['Results']

                # Set the column width for better readability
                for col_num, value in enumerate(grouped_df.columns.values):
                    max_len = max(grouped_df[value].astype(str).map(len).max(), len(value)) + 2
                    worksheet.set_column(col_num, col_num, max_len)

                # Add any additional formatting if needed
                # Example: Set header format
                header_format = workbook.add_format(
                    {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
                for col_num, value in enumerate(grouped_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

            print(f"Beautified Excel file saved as {output_file}")
        else:
            print("The specified file is not a .tdms file.")

if __name__ =="__main__":
    main()
