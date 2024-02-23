import glob
import json
import matplotlib.pyplot as plt
import os

# Directory containing the JSON files
dir_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\JSON_data\*.json'

# Initialize lists to hold data
pose_vectors = [[] for _ in range(6)]
final_q_stars = [[] for _ in range(6)]

# Iterate through each JSON file in the directory
for file_name in glob.glob(dir_path):
    if os.path.isfile(file_name):  # Check if it's a file, not a directory
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
                # Extract pose_vector and final_q_star and append to the respective lists
                for i in range(6):
                    pose_vectors[i].append(data['pose_vector'][i])
                    final_q_stars[i].append(data['final_q_star'][i])
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    else:
        print(f"Skipped {file_name}, not a file.")


# Dimension names
dimensions = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']

# Generate plots for each dimension
for i in range(6):
    plt.figure(figsize=(10, 6))
    plt.scatter(pose_vectors[i], final_q_stars[i], alpha=0.5)
    plt.title(f'{dimensions[i]}: pose_vector vs. final_q_star')
    plt.xlabel('pose_vector')
    plt.ylabel('final_q_star')
    plt.grid(True)
    plt.show()
