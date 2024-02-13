import numpy as np
import open3d as o3d
# from circle_pcd import sample_points_from_parametric_circle

def create_spheres_from_point_cloud(point_cloud, radius_scale=1.0):
    """
    Create spheres from a point cloud where each sphere is centered at a point in the point cloud.
    The radius of each sphere is scaled based on the total number of points in the point cloud.
    """
    spheres = []
    for point in point_cloud.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius_scale / len(point_cloud.points))
        sphere.translate(point)
        spheres.append(sphere)
    return spheres



if __name__ == "__main__":

    # Define the range of angle values (theta) to sample
    num_points = 100
    theta_vals = np.linspace(0, 2*np.pi, num_points)

    # Define the range of radius values to sample from
    num_radii = 20
    max_radius = 1.0
    radius_vals = np.linspace(0, max_radius, num_radii)

    # Sample points from the parametric circle
    points = sample_points_from_parametric_circle(theta_vals, radius_vals)


    # Create a point cloud from the sampled points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    radius_scale = 50

    spheres = create_spheres_from_point_cloud(point_cloud, radius_scale)

    # Visualize the spheres
    o3d.visualization.draw_geometries([point_cloud] + spheres, window_name='Spheres from Point Cloud')
