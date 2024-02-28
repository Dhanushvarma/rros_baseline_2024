import glob
import json
import matplotlib.pyplot as plt
import os

# Directory containing the JSON files
dir_path = '/home/lm-2023/Documents/rros_baseline_2024/data/csv_data_paper/JSON_(iter:500)_(depth:5)_(CCP:2)'
json_path = '/*.json'
# Initialize lists to hold data
pose_vectors = [[] for _ in range(6)]
final_q_stars = [[] for _ in range(6)]

# Iterate through each JSON file in the directory
for file_name in glob.glob(dir_path+json_path):
    if os.path.isfile(file_name):  # Check if it's a file, not a directory
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
                # Extract pose_vector and final_q_star and append to the respective lists
                for i in range(6):
                    if i in range(3,6):
                        #NOTE(dhanush) : Silly fix for final pose vector data processing
                        pose_vectors[i].append(data['pose_vector'][i]/1000)
                    else:
                        pose_vectors[i].append(data['pose_vector'][i])    
                    
                    
                    final_q_stars[i].append(data['final_q_star'][i])
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    else:
        print(f"Skipped {file_name}, not a file.")


# Dimension names
dimensions = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']

for i in range(6):
    plt.figure(figsize=(10, 6))
    plt.scatter(pose_vectors[i], final_q_stars[i], alpha=0.5)
    plt.title(f'{dimensions[i]}: pose_vector vs. final_q_star')
    plt.xlabel('pose_vector')
    plt.ylabel('final_q_star')

    # Find reasonable limits for the axes (extend slightly beyond data if needed)
    min_val = min(min(pose_vectors[i]), min(final_q_stars[i]))
    max_val = max(max(pose_vectors[i]), max(final_q_stars[i]))
    plt.xlim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
    plt.ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

    # Plot the y=x line
    plt.plot(plt.xlim(), plt.ylim(), ls='--', color='gray')  

    plt.grid(True)

    # Save the plot
    plot_filename = f'{dimensions[i]}.png'  # Construct the filename
    plot_filepath = os.path.join(dir_path, plot_filename)  # Path within the JSON directory
    plt.savefig(plot_filepath)

    plt.close()  # Close the plot before generating the next one