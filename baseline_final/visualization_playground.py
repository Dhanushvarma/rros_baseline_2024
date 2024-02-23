'''
THIS SCRIPT IS MEANT FOR VISUALIZATION OF STL FILES
'''
import numpy

import open_3d_utils as O3D
import open3d as o3d

hole_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_hole v1_rev.stl'
peg_stl_file_path = r'C:\Users\Dhanush\PycharmProjects\rros_baselines_2024\assets\stl_kuka\square_peg_origin_at_tip v1.stl'


peg_pcd = O3D.sample_points_from_stl(peg_stl_file_path, 25000)
hole_pcd = O3D.sample_points_from_stl(hole_stl_file_path, 25000)
print("PCD done !")
peg_points = numpy.asarray(peg_pcd.points)
hole_points = numpy.asarray(hole_pcd.points)

import pdb;

peg_pcd = O3D.filter_pcd_by_z(pcd=peg_pcd, z_range=[0, 3])
hole_pcd = O3D.filter_pcd_by_z(pcd=hole_pcd, z_range=[-3, 0])

pdb.set_trace()

peg_sph_rad = O3D.find_closest_pair_distance_in_pcd(peg_pcd)
hole_sph_rad = O3D.find_closest_pair_distance_in_pcd(hole_pcd)

print('peg_sph_rad:', peg_sph_rad)
print('hole_sphere_rad', hole_sph_rad)

# O3D.render_spheres_for_pcd(pcd=peg_pcd, radius=peg_sph_rad)
# O3D.render_spheres_for_pcd(pcd=hole_pcd, radius=hole_sph_rad)

O3D.render_spheres_for_pcds(pcds = [peg_pcd, hole_pcd], radii=[.5, .5], colors=[(1, 0, 0), (0, 1, 0)])

# SANITY CHECK PERFORMED
# (Pdb) peg_points[:, -1].min()
# 0.0
# (Pdb) peg_points[:, -1].min()
# 50.20000076293945
# (Pdb) hole_points[:, -1].min()
# -40.0
# (Pdb) hole_points[:, -1].max()
# 0.0
