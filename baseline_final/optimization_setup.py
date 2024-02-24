'''
THIS SCRIPT CONTAINS THE FUNCTIONS FOR THE IMPLEMENTATION OF THE OPTIMIZATION PROBLEMs
'''

import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
import pdb


# Define the function
def const_first_term(aux_var, radius, obj1_point, obj2_point):
    obj1_point = np.array(obj1_point).reshape(3, 1)  # Ensure obj1_point is correctly shaped
    obj2_point = np.array(obj2_point).reshape(3, 1)  # Ensure obj2_point is correctly shaped

    # Ensure aux_var is a NumPy array (if not already)
    aux_var = np.array(aux_var, dtype=float).reshape(-1, 1)  # Ensure aux_var is correctly shaped

    rot_matrix = R.from_euler('zyx', [aux_var[0][0], aux_var[1][0], aux_var[2][0]], degrees=False).as_matrix()

    # Correctly forming the translation vector
    translation = np.array([aux_var[3][0], aux_var[4][0], aux_var[5][0]]).reshape(3, 1)

    # Apply rotation and translation
    obj1_point_transformed = rot_matrix @ obj1_point + translation
    delta_p = obj1_point_transformed - obj2_point

    e_term = rot_matrix @ obj2_point + delta_p
    result = e_term.transpose() @ e_term

    # Subtracting the squared radius, adjusted for shape
    return result - ((2 * radius) ** 2)

def transform_object_points(obj_points, q_star):
  """
  Transforms object points using the given rotation and translation parameters.

  Args:
      obj_points: A numpy array of shape (num_points, 3) representing the object points.
      q_star: A numpy array of shape (6, 1) containing the rotation and translation parameters.

  Returns:
      A numpy array of shape (num_points, 3) representing the transformed object points.
  """

  # Extract rotation and translation components
  rot_matrix = R.from_euler('zyx', [q_star[0][0], q_star[1][0], q_star[2][0]], degrees=False).as_matrix()
  translation = np.array([q_star[3][0], q_star[4][0], q_star[5][0]]).reshape(3, 1)

  # Apply transformation to each point
  transformed_points = rot_matrix @ obj_points.T + translation

  return transformed_points.T

  
def const_second_term_matrix(aux_var, obj1_point, obj2_point):
    obj1_point = np.array(obj1_point).reshape(3, 1)  # Ensure obj1_point is correctly shaped
    obj2_point = np.array(obj2_point).reshape(3, 1)  # Ensure obj2_point is correctly shaped

    # Ensure aux_var is a NumPy array (if not already)
    aux_var = np.array(aux_var, dtype=float).reshape(-1, 1)  # Ensure aux_var is correctly shaped

    # Create rotation matrices
    rot_matrix = R.from_euler('zyx', [aux_var[0][0], aux_var[1][0], aux_var[2][0]], degrees=False).as_matrix()
    rot_x = R.from_euler('x', aux_var[0][0], degrees=False).as_matrix()
    rot_y = R.from_euler('y', aux_var[1][0], degrees=False).as_matrix()
    rot_z = R.from_euler('z', aux_var[2][0], degrees=False).as_matrix()

    translation = np.array([aux_var[3][0], aux_var[4][0], aux_var[5][0]]).reshape(3, 1)
    obj1_point_transformed = rot_matrix @ obj1_point + translation
    delta_p = obj1_point_transformed - obj2_point
    e_term = rot_matrix @ obj2_point + delta_p

    result = np.zeros((6, 1))  # Correct initialization of result array
    result[0, 0] = delta_p.transpose() @ rot_z.transpose() @ rot_y @ rot_x @ obj2_point
    result[1, 0] = delta_p.transpose() @ rot_z @ rot_y.transpose() @ rot_x @ obj2_point
    result[2, 0] = delta_p.transpose() @ rot_z @ rot_y @ rot_x.transpose() @ obj2_point
    result[3:, :] = e_term  # Correctly assign e_term to the last 3 rows

    return result


def create_contraint(first_term, second_term_matrix, aux_var, optim_var):

    aux_var_cvx = cp.Parameter((6, 1), value=aux_var)
    second_term_matrix_cvx = cp.Parameter((6, 1), value=second_term_matrix)

    expression = first_term + cp.sum(cp.multiply(second_term_matrix_cvx, (optim_var - aux_var_cvx)))

    constraint = [expression >= 0]

    return constraint


def constraint_single_pair(optim_var, aux_var, obj1_point, obj2_point, radius):

    first_term = const_first_term(aux_var=aux_var, obj1_point=obj1_point, obj2_point=obj2_point, radius=radius)
    second_term_matrix = const_second_term_matrix(aux_var=aux_var, obj1_point=obj1_point, obj2_point=obj2_point)
    constraint = create_contraint(first_term=first_term, second_term_matrix=second_term_matrix, aux_var=aux_var, optim_var=optim_var)

    return constraint


def create_optimization_problem(q_pred, q_star, G_matrix, constraints_list):
    # Ensure q_pred is a CVXPY parameter or a numpy array
    q_pred_param = cp.Parameter(q_pred.shape, value=q_pred)

    # G_matrix should be a CVXPY parameter or a numpy array if constant
    G_param = cp.Parameter(G_matrix.shape, value=G_matrix, PSD=True)  # Assuming G_matrix is positive semi-definite

    # Define the objective function
    # For a quadratic form x^T * G * x, use cp.quad_form(x, G)
    objective = cp.Minimize(0.5 * cp.quad_form(q_star - q_pred_param, G_param))

    # Create the optimization problem with the specified objective and constraints
    problem = cp.Problem(objective, constraints_list)

    return problem



if __name__ == '__main__':
    # EXAMPLE USAGE
    aux_var = np.array([[0], [0], [0], [1], [2], [3]])  # Solution from previous optimization
    radius = 1
    obj1_point = np.array([0, 0, 0])  # Origin
    obj2_point = np.array([1, 1, 1])  # A point in space

    q_star = cp.Variable((6, 1))
    G_matrix = np.eye(6)
    q_pred = np.random.rand(6, 1)

    # Generating the constraints
    result = const_first_term(aux_var, radius, obj1_point, obj2_point)
    result_2 = const_second_term_matrix(aux_var, obj1_point, obj2_point)
    constraint = constraint_single_pair(optim_var=q_star, aux_var=aux_var, obj1_point=obj1_point, obj2_point=obj2_point, radius=radius)


    problem = create_optimization_problem(q_pred=q_pred, q_star=q_star, G_matrix=G_matrix, constraints_list=constraint)

    problem.solve()

    pdb.set_trace()
    print(q_star.value)
