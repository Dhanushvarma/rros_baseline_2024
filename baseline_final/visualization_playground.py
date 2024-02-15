'''
THIS SCRIPT IS MEANT FOR VISUALIZATION OF STL FILES
'''

import open_3d_utils as O3D
import open3d as o3d

peg_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_peg_flat v1.stl'
hole_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl\cross_hole_flat v1.stl'


peg_pcd = O3D.sample_points_from_stl(peg_stl_file_path, 400)
hole_pcd = O3D.sample_points_from_stl(hole_stl_file_path, 400)


peg_sph_rad = O3D.find_closest_pair_distance_in_pcd(peg_pcd)
hole_sph_rad = O3D.find_closest_pair_distance_in_pcd(hole_pcd)

print('peg_sph_rad:', peg_sph_rad)
print('hole_sphere_rad', hole_sph_rad)


O3D.render_spheres_for_pcd(pcd=peg_pcd, radius=peg_sph_rad)




