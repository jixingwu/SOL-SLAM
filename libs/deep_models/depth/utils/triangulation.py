import numpy as np
import cv2


def convert_sparse3D_to_depth(kp, XYZ, height, width):
    depth = np.zeros((height, width))
    kp_int = kp.astype(int)
    
    y_idx = (kp_int[:, 0] >= 0) * (kp_int[:, 0] < width)
    kp_int = kp_int[y_idx]
    x_idx = (kp_int[:, 1] >= 0) * (kp_int[:, 1] < height)
    kp_int = kp_int[x_idx]
    
    XYZ = XYZ[:, y_idx]
    XYZ = XYZ[:, x_idx]
    depth[kp_int[:, 1], kp_int[:, 0]] = XYZ[2]
    
    return depth

def triangulation(inv_K, shape, kp1, kp2, T_1w, T_2w):
    """ Triangulation to get 3D points """
    
    # triangulation
    img_h, img_w, _ = shape
    ones = np.ones_like(kp1[:, :1]) # [N,1]
    kp1_norm = (inv_K @ np.hstack((kp1.copy(), ones)).T) # [3,N]
    kp2_norm = (inv_K @ np.hstack((kp2.copy(), ones)).T)
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_norm[:2], kp2_norm[:2])
    X /= X[3]
    X1_tri = T_1w[:3] @ X
    X2_tri = T_2w[:3] @ X
    
    depth1 = convert_sparse3D_to_depth(kp1, X1_tri, img_h, img_w)
    depth2 = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
    
    return depth1, depth2
    
    
     