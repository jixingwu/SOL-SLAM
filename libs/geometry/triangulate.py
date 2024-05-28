from libs.geometry.ops_3d import *
from libs.general.utils import *
import numpy as np
from triangulation import triangulate
from libs.geometry.camera_modules import SE3

def compute_sparse_depth(ref_kp, cur_kp, rel_pose, shape, cam_intrinsics, visualize=False, method='triangulation'):
    """Compute sparse depth
    """
    img_h, img_w = shape
    depth2_tri = np.zeros((img_h, img_w))
    kp1_norm = ref_kp.copy()
    kp2_norm = cur_kp.copy()

    kp1_norm[:, 0] = \
        (ref_kp[:, 0] - cam_intrinsics.cx) / cam_intrinsics.fx
    kp1_norm[:, 1] = \
        (ref_kp[:, 1] - cam_intrinsics.cy) / cam_intrinsics.fy
    kp2_norm[:, 0] = \
        (cur_kp[:, 0] - cam_intrinsics.cx) / cam_intrinsics.fx
    kp2_norm[:, 1] = \
        (cur_kp[:, 1] - cam_intrinsics.cy) / cam_intrinsics.fy

    if method == 'triangulation':
        _, _, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), rel_pose.inv_pose)

        # Triangulation outlier removal
        depth2_tri = convert_sparse3D_to_depth(cur_kp, X2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0
    elif method == 'midpoint':
        triangulated_point = np.zeros((len(kp1_norm), 3))
        for i in range(len(kp1_norm)):
            features = [kp1_norm[i], kp2_norm[i]]
            poses = [SE3.from_matrix(np.eye(4)), SE3.from_matrix(rel_pose.inv_pose)]
            triangulated_point[i] = triangulate(features, poses, algorithm="midpoint")

    # plot depth2_tri
    if visualize:
        valid_mask = (depth2_tri > 0)
        x_coords, y_coords = np.where(valid_mask)
        z_coords = depth2_tri[valid_mask]
        plt.scatter(y_coords, img_h - x_coords, c=z_coords, cmap='viridis', marker='o')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.show()

    return depth2_tri
