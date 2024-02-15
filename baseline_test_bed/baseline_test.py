import open3d as o3d
from baseline_utils import *
from baseline_final.circle_pcd import sample_points_from_parametric_circle_v2
from scipy.io import savemat

# import pdb; pdb.set_trace()


# ORIGIN is (0, 0, 0) and the translation that happens is wrt Open3D frame
# These points correspond to when both objects are aligned (0, 0, 0) - respective object centers

# FIRST INIT CONSTRAINT, AT FIRST CONTACT
def const_func_single_pair_t0(pcd_obj_1, pcd_obj_2, radius_obj_1, radius_obj_2, q_star_0):
    transformation_matrix = euler_translation_to_matrix(q_star_0[0][0], q_star_0[1][0], q_star_0[2][0], np.array([q_star_0[3][0], q_star_0[4][0], q_star_0[5][0]]))
    pcd_obj_1_transformed = pcd_obj_1.trasform(
        transformation_matrix)  # NOTE (dhanush) : We displace points in obj1 by our guess

    constraints = []  # NOTE(dhanush) : We store all the constraints here

    pcd_obj_1_points = np.array(pcd_obj_1.points)  # NOTE(dhanush): We get the points into numpy array
    pcd_obj_2_points = np.array(pcd_obj_2.points)

    for i in range(pcd_obj_1_points.shape[0]):
        # NUMBER OF POINTS IN OBJ_1

        for j in range(pcd_obj_2_points.shape[0]):
            # NUMBER OF POINTS IN OBJ_2

            constraints.append(cp.norm(pcd_obj_1_points - pcd_obj_2_points) >= (radius_obj_1 + radius_obj_2) ** 2)

    return constraints


if __name__ == '__main__':
    # Define the minimum and maximum values for each dimension of uncertainty
    min_vals = [-0.235, -0.235, -0.204, -17.6, -17.6, -3.49]  # Minimum values for each dimension
    max_vals = [0.235, 0.235, 0.204, 17.6, 17.6, 0]  # Maximum values for each dimension
    num_points_per_dim = [5, 5, 5, 10, 10, 10]  # Number of points for each dimension

    uncertainty_grid = generate_uncertainty_grid(min_vals, max_vals, num_points_per_dim)

    random_index = np.random.randint(0, uncertainty_grid.shape[0])
    delta_T = uncertainty_grid[random_index]
    # import pdb; pdb.set_trace()

    # INSTANTIATE 2 FAKE OBJECTS

    # We create two fake circle objects, just planar for testing
    num_points = 10
    num_radii = 10
    max_radius = 1.0
    theta_vals = np.linspace(0, 2 * np.pi, num_points)
    radius_vals = np.linspace(0, max_radius, num_radii)

    # RADIUS OF THE SPHERES
    sphere_radius = 1

    # Same points for both
    points_obj1 = sample_points_from_parametric_circle_v2(theta_vals, radius_vals)
    points_obj2 = sample_points_from_parametric_circle_v2(theta_vals, radius_vals)

    point_cloud_1 = o3d.geometry.PointCloud()
    point_cloud_2 = o3d.geometry.PointCloud()

    point_cloud_1.points = o3d.utility.Vector3dVector(points_obj1)
    point_cloud_2.points = o3d.utility.Vector3dVector(points_obj2)

    # Convert Open3D point clouds to numpy arrays
    points_array_1 = np.asarray(point_cloud_1.points)
    points_array_2 = np.asarray(point_cloud_2.points)

    # Save the numpy arrays to a .mat file
    savemat('point_clouds.mat', {'points_obj1': points_array_1, 'points_obj2': points_array_2})



    # obj_1_dict = {'point_cloud': point_cloud_1, 'radius': sphere_radius}
    # obj_2_dict = {'point_cloud': point_cloud_2, 'radius': sphere_radius}
    #
    # # SETUP OPTIMIZATION FOR FIRST CONTACT
    # G = np.eye(6)  # SUGGESTED BY AUTHORS
    #
    # # GET PREDICTION FOR q_0 - RANDOM SAMPLE
    # # q_0 = cp.Variable((6, 1))
    # q_0 = delta_T.reshape((6, 1))
    # q_star_0 = cp.Variable((6, 1))  # WHAT WE ARE SOLVING FOR
    #
    # # import pdb; pdb.set_trace()
    #
    # objective_t_0 = 0.5 * cp.quad_form(q_0 - q_star_0, G)
    #
    # contraints_t_0 = const_func_single_pair_t0(point_cloud_1, point_cloud_2, obj_1_dict['radius'], obj_2_dict['radius'], q_star_0)
    #
    # prob = cp.Problem(cp.Minimize(objective_t_0), contraints_t_0)
    #
    # prob.solve()




