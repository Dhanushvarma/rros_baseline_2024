'''
THIS SCRIPT IS MEANT FOR DEBUGGING THE OPTIMIZATION SETUP USING PLANAR CIRCLE DISKS
'''
import numpy as np
import cvxpy as cp
import optimization_setup as OS
from circle_pcd import *

if __name__ == '__main__':
    # EXAMPLE USAGE

    num_points = 10
    theta_vals = np.linspace(0, 2 * np.pi, num_points)

    # Define the range of radius values to sample from
    num_radii = 5
    max_radius = 1.0
    radius_vals = np.linspace(0, max_radius, num_radii)

    # Sample points from the parametric circle
    points = sample_points_from_parametric_circle_v1(theta_vals, radius_vals)

    # import pdb; pdb.set_trace()
    pcd_obj1 = o3d.geometry.PointCloud()
    pcd_obj2 = o3d.geometry.PointCloud()

    pcd_obj1.points = o3d.utility.Vector3dVector(points)
    pcd_obj2.points = o3d.utility.Vector3dVector(points)

    translation_vector = np.array([0.5, 0.5, 0.0])
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, 3] = translation_vector

    pcd_obj1.transform(transformation_matrix)  # Translating Object 1

    o3d.visualization.draw([pcd_obj1, pcd_obj2])

    # Replace with actual obj1_points
    obj1_points = np.asarray(pcd_obj1.points)
    obj2_points = np.asarray(pcd_obj2.points)

    min_vals = [-0.235, -0.235, -0.204, -17.6, -17.6, -3.49]
    max_vals = [0.235, 0.235, 0.204, 17.6, 17.6, 0]
    num_points_per_dim = [5, 5, 5, 10, 10, 10]


    UC_grid = OU.generate_uncertainty_grid(min_vals, max_vals, num_points_per_dim)
    # delta_T = UC_grid[np.random.randint(0, UC_grid.shape[0])]
    delta_T = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0])

    # OPTIMIZATION SETUP
    radius = .01 # Radius of sphere
    aux_var = np.array([[0], [0], [0], [0], [0], [0]])
    q_star = cp.Variable((6, 1))
    G_matrix = np.eye(6)
    q_pred = delta_T.reshape(6,1) # PREDICTION FROM UC GRID
    cumulative_constraints_list = []

    for i in range(obj1_points.shape[0]):

        for j in range(obj2_points.shape[0]):

            cumulative_constraints_list += OS.constraint_single_pair(optim_var=q_star, aux_var=aux_var,
                                                                     obj1_point=obj1_points[i].reshape(3,1),
                                                                     obj2_point=obj1_points[j].reshape(3,1),
                                                                     radius=radius)

            if j % 100 == 0:
                print("Still in the process of adding constraints !")


    problem = OS.create_optimization_problem(q_pred=q_pred, q_star=q_star,
                                             G_matrix=G_matrix,
                                             constraints_list=cumulative_constraints_list)

    problem.solve()

    print(problem.status)
    print(q_star.value)











