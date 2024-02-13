import numpy as np



def generate_uncertainty_grid(min_vals, max_vals, num_points_per_dim):
    """
    Generates a grid of uncertainty points covering a specified range with a specified number of points per dimension.
    """
    num_dims = len(min_vals)
    grids = []
    for i in range(num_dims):
        grid = np.linspace(min_vals[i], max_vals[i], num_points_per_dim[i])
        grids.append(grid)
    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.vstack([axis.flatten() for axis in mesh]).T
    return points








