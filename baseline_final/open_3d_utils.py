'''
UTILITY FILE | CONTAINS FUNCTION FOR OPEN 3D VISUALIZATION & OTHER STUFF
'''
import numpy as np
import open3d as o3d
import trimesh


def find_closest_pair_distance_in_pcd(pcd):
    """
    Finds the minimum distance between any two points in a point cloud.

    Args:
    - pcd: An Open3D PointCloud object.

    Returns:
    - The minimum distance between any two points in the point cloud.
    """
    # Convert the Open3D point cloud to a numpy array if it's not already
    points = np.asarray(pcd.points)

    # Build a KDTree for the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    min_distance = float('inf')  # Initialize minimum distance with infinity

    # Iterate through each point in the point cloud
    for i in range(len(points)):
        # Use the KDTree to find the 2 nearest neighbors of the point
        # The first nearest neighbor will be the point itself, so we ask for 2 neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)

        # Ensure at least 2 points are found (the point itself and one neighbor)
        if k >= 2:
            # The second point (index 1) is the closest other point to the current point
            nearest_point = points[idx[1]]
            distance = np.linalg.norm(points[i] - nearest_point)

            # Update the minimum distance if this distance is smaller
            min_distance = min(min_distance, distance)

    return min_distance



def collect_leaf_node_origins(node, node_info, leaf_origins):
    """
    Modified traverse function to collect leaf node origins.

    Args:
    - node: The current octree node being traversed.
    - node_info: Information about the current node (depth, origin, etc.).
    - leaf_origins: List to accumulate the origins of leaf nodes.
    """
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        # For leaf nodes, add the origin to the list
        leaf_origins.append(node_info.origin)
    # No early stopping criterion in this case; we want to visit all nodes
    return False  # Continue traversal


def create_spheres_at_origins(leaf_origins, radius=0.05):
    """
    Create sphere meshes at the collected origins.

    Args:
    - leaf_origins: List of origins collected from the octree traversal.
    - radius: Radius of each sphere.

    Returns:
    A list of sphere meshes.
    """
    spheres = []
    for origin in leaf_origins:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(origin)
        spheres.append(sphere)
    return spheres






def stl_to_point_cloud(stl_file_path, number_of_points=10000):
    """
    Convert an STL file to a point cloud.

    Parameters:
    - stl_file_path: Path to the STL file.
    - number_of_points: Number of points to sample on the mesh surface.

    Returns:
    - pcd: The resulting point cloud.
    """
    # Load the STL file
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.compute_vertex_normals()  # Optional: Compute vertex normals if needed for visualization

    # Check if the mesh is watertight. If not, the conversion might not be accurate.
    if not mesh.is_watertight():
        print("Warning: Mesh is not watertight. Consider repairing the mesh for accurate point sampling.")

    # Convert the mesh to a point cloud by sampling points on the mesh surface
    pcd = mesh.sample_points_poisson_disk(number_of_points)

    return pcd


def point_cloud_to_octree(pcd, octree_depth=12):
    """
    Convert a point cloud to an octree.

    Parameters:
    - pcd: The point cloud to convert.
    - octree_depth: The maximum depth of the octree.

    Returns:
    - octree: The resulting octree.
    """
    # Create an octree at the specified depth
    octree = o3d.geometry.Octree(max_depth=octree_depth)

    # Convert the point cloud to an octree
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    return octree


def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 1
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def render_spheres_for_pcd(pcd, radius=0.05):
    """
    Renders spheres at the locations of points in a point cloud with a specified radius.

    Args:
    - pcd: An Open3D PointCloud object.
    - radius: The radius of the spheres to be rendered at each point location.
    """
    # Ensure pcd is a point cloud object
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError("pcd must be an instance of open3d.geometry.PointCloud")

    # Extract points from the point cloud
    points = np.asarray(pcd.points)

    # Create a sphere mesh for each point
    spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(point) for point in points]

    # Compute normals for better lighting in visualization
    for sphere in spheres:
        sphere.compute_vertex_normals()

    # Visualize the spheres
    o3d.visualization.draw_geometries(spheres, window_name="Spheres at Point Cloud Locations")


def sample_points_from_stl_fancy(stl_path, number_of_points=1000):
    # Load the STL file
    mesh = trimesh.load_mesh(stl_path)

    # Initially, attempt to sample a larger number of points to account for potential duplicates
    initial_sample_size = int(number_of_points * 1.1)  # Increase by 10% as a starting point

    # Placeholder for the unique points sampled
    unique_points = np.array([]).reshape(0, 3)  # Starting with an empty array
    attempts = 0  # To avoid infinite loops in edge cases

    while unique_points.shape[0] < number_of_points and attempts < 10:
        # Sample points uniformly from the mesh surface
        sampled_points = trimesh.sample.sample_surface_even(mesh, count=initial_sample_size)[0]

        # Find unique points to avoid duplicates
        _, unique_indices = np.unique(sampled_points, axis=0, return_index=True)
        unique_sampled_points = sampled_points[unique_indices]

        # Combine with previously found unique points (if this is a repeat attempt)
        unique_points = np.unique(np.vstack((unique_points, unique_sampled_points)), axis=0)

        # Increase sample size for the next iteration if necessary
        initial_sample_size += int(number_of_points * 0.1)  # Increasing by 10% of the original target each time
        attempts += 1

    # If we have more than the desired number of points, trim the array
    if unique_points.shape[0] > number_of_points:
        unique_points = unique_points[:number_of_points]

    # Convert to Open3D point cloud for further processing or visualization
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(unique_points)

    return point_cloud


def sample_points_from_stl(stl_path, number_of_points=1000):
    # Load the STL file
    mesh = trimesh.load_mesh(stl_path)

    # Sample points uniformly
    # The sample method returns points uniformly distributed over the mesh surface
    # points = mesh.sample(number_of_points)
    points = trimesh.sample.sample_surface_even(mesh, count=number_of_points)

    # Convert to Open3D point cloud for further processing or visualization
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[0])

    return point_cloud