import pandas as pd
import numpy as np


def add_noise_and_clamp(values, noise_variance, clamp_min, clamp_max):
    """Add normal noise to the values and clamp the results within specified bounds."""
    noise = np.random.normal(0, noise_variance, len(values))
    return np.clip(values + noise, clamp_min, clamp_max)


def process_csv(input_csv_path, output_csv_path):
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Add noise for Tx, Ty, Tz with variance 1, clamped between -0.01 and 0.01
    df['Tx'] = add_noise_and_clamp(df['roll'], 1, -0.01, 0.01)
    df['Ty'] = add_noise_and_clamp(df['pitch'], 1, -0.01, 0.01)
    df['Tz'] = add_noise_and_clamp(df['yaw'], 1, -0.01, 0.01)

    # Add noise for Fx, Fy, Fz with variance 1, clamped between -2 and 2
    df['Fx'] = add_noise_and_clamp(df['x'], 1, -2, 2)
    df['Fy'] = add_noise_and_clamp(df['y'], 1, -2, 2)
    df['Fz'] = add_noise_and_clamp(df['z'], 1, -2, 2)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)


# Example usage
input_csv_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\csv_data_paper\roll_pitch_yaw_xyz_formatted.csv'  # Replace with your actual input CSV file path
output_csv_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\csv_data_paper\paper_data_force_noise.csv'  # Replace with your desired output CSV file path

# Uncomment the following line to run the function with your file paths
process_csv(input_csv_path, output_csv_path)
