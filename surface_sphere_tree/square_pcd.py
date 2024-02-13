import numpy as np
import open3d as o3d

# Define the vertices of the square (assuming it's centered at the origin)
square_vertices = np.array([
    [-1, -1, 0],  # Bottom-left corner
    [1, -1, 0],   # Bottom-right corner
    [1, 1, 0],    # Top-right corner
    [-1, 1, 0]    # Top-left corner
])

# Define the number of points to sample along each side of the square
num_points_per_side = 10

# Initialize an empty array to store sampled points
sampled_points = []

# Sample points uniformly from a grid on the square's surface
for i in range(num_points_per_side):
    for j in range(num_points_per_side):
        # Compute the coordinates of the point within the square
        u = -1 + (2 / (num_points_per_side - 1)) * i
        v = -1 + (2 / (num_points_per_side - 1)) * j
        point = np.array([u, v, 0])
        sampled_points.append(point)

# Create a point cloud from the sampled points
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud], window_name='Sampled Points from Square Surface')
