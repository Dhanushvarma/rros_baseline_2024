import numpy as np
import open3d as o3d

import numpy as np

def sample_points_from_parametric_circle_v1(theta_vals, radius_vals):
    """
    Samples points uniformly from a parametric circle defined by varying radius values.
    theta_vals are array of angle values in radians.
    """
    points = np.zeros((len(theta_vals) * len(radius_vals), 3))
    idx = 0
    for r in radius_vals:
        for theta in theta_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points[idx] = np.array([x, y, 0])
            idx += 1
    return points


def sample_points_from_parametric_circle_v2(theta_vals, radius_vals):
    """
    Samples points uniformly from a parametric circle defined by varying radius values.
    theta_vals are array of angle values in radians.
    """
    points = np.zeros((len(theta_vals) * len(radius_vals), 3))
    idx = 0
    for r in radius_vals:
        for theta in theta_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points[idx] = np.array([x, y, 0])
            idx += 1
    return points





if __name__ == '__main__':
    # Example usage:
    # radius_range = (0., 1.0)  # Example: sample radii from 0.5 to 1.0
    # num_points = 100
    # theta_vals = np.linspace(0, 2 * np.pi, num_points)
    # points_on_circle_and_interior = sample_points_from_parametric_circle_v1(theta_vals, radius_range)
    #
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points_on_circle_and_interior)
    #
    # o3d.visualization.draw_geometries([point_cloud], window_name='Sampled Points from Parametric Triangle')

    # Define the range of angle values (theta) to sample
    num_points = 100
    theta_vals = np.linspace(0, 2 * np.pi, num_points)

    # Define the range of radius values to sample from
    num_radii = 20
    max_radius = 1.0
    radius_vals = np.linspace(0, max_radius, num_radii)

    # Sample points from the parametric circle
    points = sample_points_from_parametric_circle_v2(theta_vals, radius_vals)

    import pdb; pdb.set_trace()
    # Create a point cloud from the sampled points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], window_name='Sampled Points from Parametric Circle')



