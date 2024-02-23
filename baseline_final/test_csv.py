import numpy as np
import cvxpy as cp
import pandas as pd

file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\kuka_pose_wrench_benchmark_frame.csv'

def quaternion_to_euler(row):
    # Extract quaternion components from the row
    qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']

    # Conversion formula
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return pd.Series([roll, pitch, yaw])


df = pd.read_csv(file_path)

# Apply the conversion function to each row
euler_angles = df.apply(quaternion_to_euler, axis=1)

# The resulting DataFrame has columns for roll, pitch, and yaw
euler_angles.columns = ['roll', 'pitch', 'yaw']

# Optionally, you can append these Euler angles back to your original DataFrame
df = pd.concat([df, euler_angles], axis=1)

# Display the first few rows of the DataFrame
print(df.head())

output_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\kuka_data_euler.csv'

df.to_csv(output_file_path, index=False)