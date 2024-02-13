from open_3d_utils import *

# MAKE THE PCD FROM STL FILES

#TRANSlATE THE PCD FILES to make them collide

# Function which loops through the PCD's and check if they are less than (radius_pcd_1 + radius_pcd_2)
# This should check between two PCD's and not within the PCD obviosuly
# then render both the PCD's as spheres using where the colliding points are rendered "red" and the other normal
# This function should take as arguement the diamter to use for the spheres for the PCD's in the above line

def render_collision(pcd1, pcd2, diameter1, diameter2):
    # Extract points
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    radius1 = diameter1 / 2
    radius2 = diameter2 / 2

    # KDTree for efficient collision checking
    pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
    pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

    # Collision checking is done by expanding search radius by the sum of both radii
    collision_indices1 = set()
    collision_indices2 = set()

    # Check collisions from pcd1 to pcd2
    for i, point in enumerate(points1):
        [k, idx, _] = pcd_tree2.search_radius_vector_3d(point, radius1 + radius2)
        if k > 0:  # Collision detected
            collision_indices1.add(i)
            collision_indices2.update(idx)

    # Check collisions from pcd2 to pcd1 (to catch any missed points)
    for i, point in enumerate(points2):
        [k, idx, _] = pcd_tree1.search_radius_vector_3d(point, radius1 + radius2)
        if k > 0:  # Collision detected
            collision_indices2.add(i)
            collision_indices1.update(idx)

    # Create spheres for PCD1 and PCD2 with respective colors and radii
    spheres = []
    for i, point in enumerate(points1):
        color = [1, 0, 0] if i in collision_indices1 else [0, 1, 0]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius1)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    for i, point in enumerate(points2):
        color = [1, 0, 0] if i in collision_indices2 else [0, 0, 1]  # Using blue for the second PCD
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius2)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    # Visualize
    o3d.visualization.draw_geometries(spheres, window_name="Collision Visualization")



stl_file_path = r"C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_peg v6.stl"
stl_file_path_2 = r""
octree_depth = 6  # Adjust based on your requirements
number_of_points = 1000  # Adjust based on the desired density of the point cloud
pcd_1 = stl_to_point_cloud(stl_file_path, number_of_points)
pcd_2 = stl_to_point_cloud(stl_file_path, number_of_points)

diameter_1 = find_closest_pair_distance_in_pcd(pcd_1)
diameter_2 = find_closest_pair_distance_in_pcd(pcd_2)

translation_vector = np.array([0, 0, 50])
rotation_angles = np.array([np.radians(0), np.radians(180), np.radians(180)])
rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)

pcd_2.translate(translation_vector, relative=True)
pcd_2.rotate(rotation_matrix, center=pcd_2.get_center())

render_collision(pcd_1, pcd_2, diameter_1, diameter_2)

