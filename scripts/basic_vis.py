from open_3d_utils import *



# Specify your STL file path and octree parameters here
stl_file_path = r"C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_peg v6.stl"
stl_file_path_2 = r"C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_hole2.stl"
octree_depth = 6  # Adjust based on your requirements
number_of_points = 100  # Adjust based on the desired density of the point cloud


pcd = stl_to_point_cloud(stl_file_path, number_of_points)
pcd_2 = stl_to_point_cloud(stl_file_path_2, number_of_points)

# ------------------ #



# Assuming pcd1 is the point cloud you want to translate
pcd.translate(translation_vector, relative=True)
pcd.rotate(rotation_matrix, center=pcd.get_center())

# ------------------ #
octree = point_cloud_to_octree(pcd, octree_depth)
octree_2 = point_cloud_to_octree(pcd_2, octree_depth)

# Visualize the octree
# o3d.visualization.draw([octree, octree_2])

octree.traverse(f_traverse)


