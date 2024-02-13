from open_3d_utils import *

stl_file_path = r"C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_peg v6.stl"
octree_depth = 6  # Adjust based on your requirements
number_of_points = 100  # Adjust based on the desired density of the point cloud
pcd = stl_to_point_cloud(stl_file_path, number_of_points)
octree = point_cloud_to_octree(pcd, octree_depth)


# Assuming `octree` is your octree object
leaf_origins = []
octree.traverse(lambda node, node_info: collect_leaf_node_origins(node, node_info, leaf_origins))

diameter = find_closest_pair_distance_in_pcd(pcd)
radius = diameter/2

# Create spheres at the collected origins
spheres = create_spheres_at_origins(leaf_origins, radius=radius)  # Adjust radius as needed

# Visualize
o3d.visualization.draw(spheres)


