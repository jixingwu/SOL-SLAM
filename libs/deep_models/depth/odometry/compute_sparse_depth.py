import cv2

def compute_sparse_depth(img1, img2, relative_pose):
    """ 计算稀疏深度图
    1. 利用opencv 计算img1和img2的光流点
    2. 利用相对位姿计算光流点的深度值
    """
    pass