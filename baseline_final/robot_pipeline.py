'''
THIS SCRIPT IS THE MAIN SCRIPT FOR THE ROBOT PIPELINE, USE SIMILAR FASHION IN THE ROS CODE
'''

import sys
import numpy as np
import pandas as pd
import cvxpy as cp
import optimization_setup as OS
import open_3d_utils as O3D
from circle_pcd import *


class robot_pipeline:

    def __init__(self):
        self.min_vals_UC = None  # MIN VALUES FOR THE UNCERTAINTY GRID
        self.max_vals_UC = None  # MAX VALUES FOR THE UNCERTAINTY GRID

        self.q_star_init = np.array([[0], [0], [0], [0], [0], [0]])  # q_star for t = -1 timestep

        self.obj1_sph_rad = None  # OBJECT 1 PCD SPHERE RADIUS
        self.obj2_sph_rad = None  # OBJECT 2 PCD SPHERE RADIUS
        self.stiffness_matrix = None  # STIFFNESS MATRIX - set to identity

        self.obj1_pcd = None  # PCD OF OBJECT 1
        self.obj2_pcd = None  # PCD OF OBJECT 2
        self.obj1_points = None  # POINTS OF OBJECT 1
        self.obj2_points = None  # POINTS OF OBJECT 2

        self.force_sensed = None  # INPUT FORCE SENSED IN REAL LIFE


    def set_params(self, stiffness, obj1_sphere_rad, obj2_sphere_rad, min_vals_UC, max_vals_UC):
        self.stiffness_matrix = stiffness
        self.obj1_sph_rad = obj1_sphere_rad
        self.obj2_sph_rad = obj2_sphere_rad
        self.min_vals_UC = min_vals_UC
        self.max_vals_UC = max_vals_UC

        return None

    def set_pcd_objects(self, obj1_pcd, obj2_pcd):
        self.obj1_pcd = obj1_pcd
        self.obj2_pcd = obj2_pcd

        self.obj1_points = np.asarray(self.obj1_pcd.points)
        self.obj2_points = np.asarray(self.obj2_pcd.points)

        return None

    def sample_UC_grid(self):

        return np.random.uniform(np.array(self.min_vals_UC), np.array(self.max_vals_UC)).reshape(6,1)

    def set_force_sensed(self, force_sensed):
        self.force_sensed = force_sensed

        return None

    def give_constraints_list(self, q_star, aux_var):

        constraints_list = []

        for i in range(self.obj1_points.shape[0]):

            for j in range(self.obj2_points.shape[0]):

                constraints_list += OS.constraint_single_pair(optim_var= q_star, aux_var=aux_var,
                                                              obj1_point=self.obj1_points[i].reshape(3, 1),
                                                              obj2_point=self.obj2_points[j].reshape(3, 1),
                                                              radius=self.obj1_sph_rad)

        return constraints_list

    def opt_problem(self, q_star, q_pred, aux_var):

        constraints_list = self.give_constraints_list(q_star, aux_var)

        problem = OS.create_optimization_problem(q_pred= q_pred, q_star= q_star,
                                                 G_matrix=self.stiffness_matrix,
                                                 constraints_list=constraints_list)

        return problem, q_star


def read_data_from_csv(csv_file_path, row_number):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    force_sensed = df.loc[row_number, ['Tx', 'Ty', 'Tz', 'Fx', 'Fy', 'Fz']].to_numpy().reshape(6, 1)
    position_vector_mm = df.loc[row_number, ['x', 'y', 'z']].to_numpy() * 1000
    orientation_vector = df.loc[row_number, ['roll', 'pitch', 'yaw']].to_numpy()

    pose_vector = np.vstack((orientation_vector, position_vector_mm)).reshape(6, 1)

    return force_sensed, pose_vector


if __name__ == '__main__':

    # NOTE(dhanush) : These are the file paths
    csv_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\data\kuka_data_euler.csv'
    hole_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_hole v1_rev.stl'
    peg_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_peg_origin_at_tip v1.stl'

    # NOTE(dhanush) : Data Selection Manually
    row_number = 0
    force_sensed, pose_vector = read_data_from_csv(csv_file_path, row_number)
    print("Pose Vector from CSV:", pose_vector.T)
    print("Force Vector from CSV:", force_sensed.T)

    # UNCERTAINTY GRID - continuous
    min_vals = [-0.235, -0.235, -0.204, -17.6, -17.6, -3.49]
    max_vals = [0.235, 0.235, 0.204, 17.6, 17.6, 0]

    # NOTE: HERE WE INPUT THE NUMBER OF POINTS TO HAVE FOR EACH OBJECT
    peg_pcd = O3D.sample_points_from_stl(peg_stl_file_path, 10000)  # OBJECT 1
    hole_pcd = O3D.sample_points_from_stl(hole_stl_file_path, 10000)  # OBJECT 2

    peg_pcd = O3D.filter_pcd_by_z(pcd=peg_pcd, z_range=[0, 0.0001])  # TRIMMING THE PCD
    hole_pcd = O3D.filter_pcd_by_z(pcd=hole_pcd, z_range=[-0.0001, 0])  # TRIMMING THE PCD
    print("Number of Points in the Peg : ", np.asarray(peg_pcd.points).shape[0])
    print("Number of Points in the Hole : ", np.asarray(hole_pcd.points).shape[0])

    peg_sph_rad = O3D.find_closest_pair_distance_in_pcd(peg_pcd)
    hole_sph_rad = O3D.find_closest_pair_distance_in_pcd(hole_pcd)

    sph_rad = min(peg_sph_rad, hole_sph_rad)  # TODO : Check if doing this is correct

    O3D.render_spheres_for_pcd(pcd=peg_pcd, radius=sph_rad - 0.1)  # RENDER TO CHECK
    O3D.render_spheres_for_pcd(pcd=hole_pcd, radius=sph_rad - 0.1)  # RENDER TO CHECK
    O3D.render_spheres_for_pcds(pcds=[peg_pcd, hole_pcd], radii=[.5, .5], colors=[(1, 0, 0), (0, 1, 0)])

    # Algorithm Object
    baseline_object = robot_pipeline()

    # Open3D Stuff HERE and SET PARAMS
    baseline_object.set_pcd_objects(obj1_pcd=peg_pcd, obj2_pcd=hole_pcd)
    baseline_object.set_params(stiffness=np.eye(6), obj1_sphere_rad=sph_rad - 0.1, obj2_sphere_rad=sph_rad - 0.1,
                               min_vals_UC=min_vals, max_vals_UC=max_vals)

    # INPUT SENSED FORCE HERE
    baseline_object.set_force_sensed(force_sensed=force_sensed)
    q_star = cp.Variable((6,1))  # THE OPTIMIZATION VARIABLE

    final_config = None  # TO STORE THE FINAL ANSWER FOR 1 FORCE SIGNAL
    # EPSILON BOUND FOR FORCE TERM | (F_c) - (F_s) < EPSILON
    EPSILON = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    optimization_problem_count = 0  # SAMPLES from set "U" counter
    MAX_ITERATIONS = 5  # Maximum number of iterations

    while True:

        q_pred = baseline_object.sample_UC_grid()
        print("q_pred is:", q_pred)

        optimization_problem = \
        baseline_object.opt_problem(q_star=q_star, q_pred=q_pred, aux_var=baseline_object.q_star_init)[0]
        optimization_problem.solve(solver=cp.ECOS, verbose=False)
        optimization_problem_count += 1

        print("Optimization problem Status: ", optimization_problem.status)
        print("Q star: ", q_star.value)

        # CALCULATING THE FORCE BASED ON THE DIFFERENCE
        force_calculated = baseline_object.stiffness_matrix @ (q_pred - q_star.value)

        force_difference = np.abs(baseline_object.force_sensed - force_calculated)

        if np.all(force_difference <= EPSILON.reshape(6, 1)) or optimization_problem_count >= MAX_ITERATIONS:
            final_config = q_star.value
            print(f"Problem solved in {optimization_problem_count} iterations.")
            break

        else:
            # Your condition to update q_star_init or reset the optimization variable

            # This part remains unchanged

            if q_star.value is None:  # Use 'is None' for a proper None check
                baseline_object.q_star_init = np.zeros((6, 1))
            else:
                baseline_object.q_star_init = q_star.value

            q_star = cp.Variable((6, 1))  # Reinitialize the optimization variable

    print("Optimized Configuration (q_star.value):", q_star.value.T)
    print("Pose Vector from CSV:", pose_vector.T)

    difference = q_star.value - pose_vector
    print("Difference between optimized configuration and CSV pose vector:", difference.T)


























