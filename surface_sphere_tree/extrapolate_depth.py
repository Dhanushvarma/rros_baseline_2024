import numpy as np
import open3d as o3d
from circle_pcd import sample_points_from_parametric_circle_v2
from populate_with_spheres import  create_spheres_from_point_cloud
def extrapolate_points_along_depth(points, depth_values):
    """
    Extrapolates points sampled from a surface along certain depth values.
    """
    num_depth_values = len(depth_values)
    num_points = len(points)
    extrapolated_points = np.zeros((num_points * num_depth_values, 3))

    for i in range(num_depth_values):
        depth = depth_values[i]
        start_idx = i * num_points
        end_idx = (i + 1) * num_points
        extrapolated_points[start_idx:end_idx, :2] = points[:, :2]  # Keep xy-coordinates the same
        extrapolated_points[start_idx:end_idx, 2] = depth  # Set z-coordinate to the depth value

    return extrapolated_points


num_points = 100
theta_vals = np.linspace(0, 2 * np.pi, num_points)

# Define the range of radius values to sample from
num_radii = 20
max_radius = 1.0
radius_vals = np.linspace(0, max_radius, num_radii)

# Sample points from the parametric circle
points = sample_points_from_parametric_circle_v2(theta_vals, radius_vals)


depth_values = np.linspace(0, 1, 10)
extrapolated_points = extrapolate_points_along_depth(points, depth_values)

# Create a point cloud from the extrapolated points
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(extrapolated_points)

# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud], window_name='Extrapolated Points Along Depth')


radius_scale = 1000
spheres = create_spheres_from_point_cloud(point_cloud, radius_scale)
# Visualize the spheres

import pdb; pdb.set_trace();
# o3d.visualization.draw_geocmetries([point_cloud] + spheres, window_name='Spheres from Point Cloud')