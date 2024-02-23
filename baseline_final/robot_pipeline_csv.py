import sys
import numpy as np
import pandas as pd
import cvxpy as cp
import optimization_setup as OS
import open_3d_utils as O3D
from circle_pcd import *
from robot_pipeline import robot_pipeline, read_data_from_csv
import json

if __name__ == '__main__':

    # NOTE(dhanush) : These are the file paths
    csv_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\csv_data_all\kuka_data_euler.csv'
    hole_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_hole v1_rev.stl'
    peg_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_peg_origin_at_tip v1.stl'

    # PANDAS DF for the csv file path
    df = pd.read_csv(csv_file_path)
    df = df.dropna()  # To Drop any bad rows
    num_rows = len(df)

    # UNCERTAINTY GRID - continuous range
    min_vals = [-0.235, -0.235, -0.204, -17.6, -17.6, -3.49]
    max_vals = [0.235, 0.235, 0.204, 17.6, 17.6, 0]

    # NOTE: INPUT POINTS FOR OBJECT
    peg_pcd = O3D.sample_points_from_stl(peg_stl_file_path, 5000)  # OBJECT 1
    hole_pcd = O3D.sample_points_from_stl(hole_stl_file_path, 5000)  # OBJECT 2
    peg_pcd = O3D.filter_pcd_by_z(pcd=peg_pcd, z_range=[0, 0.0001])  # TRIMMING THE PCD
    hole_pcd = O3D.filter_pcd_by_z(pcd=hole_pcd, z_range=[-0.0001, 0])  # TRIMMING THE PCD

    print("Number of Points in the Peg : ", np.asarray(peg_pcd.points).shape[0])
    print("Number of Points in the Hole : ", np.asarray(hole_pcd.points).shape[0])

    peg_sph_rad = O3D.find_closest_pair_distance_in_pcd(peg_pcd)  # Sphere Radius for Peg
    hole_sph_rad = O3D.find_closest_pair_distance_in_pcd(hole_pcd)  # Sphere Radius for Hole
    sph_rad = min(peg_sph_rad, hole_sph_rad)  # TODO : Check if doing this is correct

    # O3D.render_spheres_for_pcd(pcd=peg_pcd, radius=sph_rad - 0.1)  # RENDER TO CHECK
    # O3D.render_spheres_for_pcd(pcd=hole_pcd, radius=sph_rad - 0.1)  # RENDER TO CHECK
    # O3D.render_spheres_for_pcds(pcds=[peg_pcd, hole_pcd], radii=[.5, .5], colors=[(1, 0, 0), (0, 1, 0)])
    # import pdb; pdb.set_trace()


    # LOOPING THROUGH THE ROWS PROVIDED
    for row_number in range(num_rows):
        force_sensed, pose_vector = read_data_from_csv(csv_file_path, row_number)

        # Algorithm Object - INSTANTIATION
        baseline_object = robot_pipeline()
        # Open3D Stuff HERE and SET PARAMS
        baseline_object.set_pcd_objects(obj1_pcd=peg_pcd, obj2_pcd=hole_pcd)
        baseline_object.set_params(stiffness=np.eye(6), obj1_sphere_rad=sph_rad - 0.05, obj2_sphere_rad=sph_rad - 0.05,
                                   min_vals_UC=min_vals, max_vals_UC=max_vals)

        # INPUT SENSED FORCE HERE
        baseline_object.set_force_sensed(force_sensed=force_sensed)
        q_star = cp.Variable((6, 1))  # THE OPTIMIZATION VARIABLE
        q_star_history = []  # To store intermediate q_star values

        EPSILON = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # EPSILON BOUND FOR FORCE TERM | (F_c) - (F_s) < EPSILON
        MAX_ITERATIONS = 5  # Maximum number of iterations

        for iteration in range(MAX_ITERATIONS):
            # Sample and solve optimization problem
            q_pred = baseline_object.sample_UC_grid()
            optimization_problem = baseline_object.opt_problem(q_star=q_star, q_pred=q_pred, aux_var=baseline_object.q_star_init)[0]
            optimization_problem.solve(solver=cp.ECOS, verbose=False)

            # Store intermediate q_star value if not None
            if q_star.value is not None:
                q_star_history.append(q_star.value.flatten().tolist())

            # Calculate force difference and check for convergence
            force_calculated = baseline_object.stiffness_matrix @ (q_pred - q_star.value)
            force_difference = np.abs(force_sensed - force_calculated)

            if np.all(force_difference <= EPSILON.reshape(6, 1)) or iteration == MAX_ITERATIONS - 1:
                break
            else:
                baseline_object.q_star_init = q_star.value if q_star.value is not None else np.zeros((6, 1))

        # Prepare and save results
        results_data = {
            'force_components': {
                'Tx': force_sensed[0, 0],
                'Ty': force_sensed[1, 0],
                'Tz': force_sensed[2, 0],
                'Fx': force_sensed[3, 0],
                'Fy': force_sensed[4, 0],
                'Fz': force_sensed[5, 0]
            },
            'pose_vector': pose_vector.flatten().tolist(),  # pose_vector is a numpy array
            'final_q_star': q_star.value.flatten().tolist() if q_star.value is not None else None,
            'intermediate_q_stars': [q_star_val for q_star_val in q_star_history]  # Directly use the list items
        }

        output_file_path = f'C:\\Users\\Dhanush\\PycharmProjects\\rros_baselines_2024\\data\\csv_data_all\\output_row_{row_number}.json'

        # Save the results to a JSON file
        with open(output_file_path, 'w') as json_file:
            json.dump(results_data, json_file, indent=4)

        print(f"Output for row {row_number} saved to {output_file_path}")






