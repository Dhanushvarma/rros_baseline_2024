import numpy as np
import open3d as o3d

def sample_points_from_parametric_triangle(u_vals, v_vals, triangle_vertices):
    """
    Samples points uniformly from a parametric triangle defined by its vertices.
    u_vals and v_vals are arrays of parameter values in the range [0, 1].
    """
    u, v = np.meshgrid(u_vals, v_vals)
    mask = u + v <= 1
    u = u[mask].flatten()
    v = v[mask].flatten()
    w = 1 - u - v
    points = (u[:, np.newaxis] * triangle_vertices[0] +
              v[:, np.newaxis] * triangle_vertices[1] +
              w[:, np.newaxis] * triangle_vertices[2])
    return points

# Define the vertices of the triangle
triangle_vertices = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1, 0])]

# Define the range of parameter values
u_vals = np.linspace(0, 1, 100)
v_vals = np.linspace(0, 1, 100)

# Sample points from the parametric triangle
points = sample_points_from_parametric_triangle(u_vals, v_vals, triangle_vertices)

# Create a point cloud from the sampled points
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud], window_name='Sampled Points from Parametric Triangle')
