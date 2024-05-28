import numpy as np
import cv2
from matplotlib import pyplot as plt

class GeometryComputing:
    def __init__(self, img_tgt, img_ipt, pose_tgt, pose_ipt, K):
        self.shape = img_tgt.shape
        self.pose_target = pose_tgt
        self.pose_input = pose_ipt[-1]
        self.K = K
        # resize imgs
        self.img_target = img_tgt
        self.img_input = img_ipt[-1]

    def compute_corresponding_point(self, pts, target, input):
        # point_p: [(x1,y1), (x2,y2), ...] Nx2
        gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        gray_ipt = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        lk_params = dict(winSize=(15,15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feat_tgt = np.array(pts, dtype=np.float32).reshape(-1,1,2)
        feat_ipt, status, _ = cv2.calcOpticalFlowPyrLK(gray_tgt, gray_ipt, feat_tgt, None, **lk_params)

        good_feat_ipt = feat_ipt[status == 1]

        ## visualization
        # for pt in pts:
        #     cv2.circle(target, (int(pt[0]), int(pt[1])), 5, (0,0,255), -1)
        # for pt in good_feat_ipt:
        #     cv2.circle(input, (int(pt[0]), int(pt[1])), 5, (0,0,255), -1)

        # colors = np.hstack((target, input))
        # cv2.imshow('colors', colors)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return good_feat_ipt, status.squeeze()

    def compute_depth_value(self, tgt, ipt, status):
        ## compute sparse 3D point coordinate
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        depth = np.zeros((tgt.shape[0], 1))
        tgt = tgt[status == 1]

        tgt_norm , ipt_norm = tgt.copy(), ipt.copy()
        tgt_norm[:, 0] = (tgt[:, 0] - cx) / fx
        tgt_norm[:, 1] = (tgt[:, 1] - cy) / fy
        ipt_norm[:, 0] = (ipt[:, 0] - cx) / fx
        ipt_norm[:, 1] = (ipt[:, 1] - cy) / fy

        tgt_3D = np.ones((3, tgt.shape[0]))
        ipt_3D = np.ones((3, ipt.shape[0]))
        tgt_3D[0], tgt_3D[1] = tgt_norm[:,0].copy(), tgt_norm[:, 1].copy()
        ipt_3D[0], ipt_3D[1] = ipt_norm[:,0].copy(), ipt_norm[:, 1].copy()

        # X = cv2.triangulatePoints(pose_tgt[:3], pose_ipt[:3], tgt_3D[:2], ipt_3D[:2])
        X = cv2.triangulatePoints(np.linalg.inv(self.pose_target)[:3], np.linalg.inv(self.pose_input)[:3], tgt_3D[:2], ipt_3D[:2])
        X /= X[3]
        X_tgt = np.linalg.inv(self.pose_target)[:3] @ X #[3,N]

        depth[status==1]=X_tgt[2,:].reshape(-1,1) #[N,1]

        # return X_tgt[2,:]
        return depth, depth > 0

    # def compute_point_depth(img_target, img_input, pose_target, pose_input, point_target, K):
    # def compute_point_depth(img_tgt, img_ipt, rel_pose, intrinsics, mask, shape):
    def compute_point_depth_for_random(self, mask):
        # point_target: (Nx2)
        # ======================
        # pts_tgt = np.array([(100, 200), (200, 300), (300, 400), (400, 500), (400, 100), (400, 200)]).astype(np.float32)
        H,W,C = self.shape
        # matrix = torch.zeros((H,W,2), dtype=torch.int)
        # for i in range(H):
        #     for j in range(W):
        #         matrix[i,j,0] = i
        #         matrix[i,j,1] = j
        Mask = mask.clone().detach().cpu().squeeze().numpy()
        
        matrix = np.array([[(i,j)for j in range(W)] for i in range(H)])
        point_target = matrix[Mask]

        pts_ipt, status = self.compute_corresponding_point(point_target, self.img_target.astype(np.uint8), self.img_input.astype(np.uint8))
        tri_tgt  = self.compute_depth_value(point_target, pts_ipt, status, self.pose_target, self.pose_input, self.K)

        return tri_tgt, tri_tgt > 0
        # ======================
    
    def compute_point_depth_for_edge(self, rows, cols):
        # point_target: (Nx2);
        # rows, cols: 4xN
        H,W,C = self.shape
        mask = np.zeros(rows.shape, np.bool_)
        depth = np.zeros(rows.shape, np.float32)
        for i in range(rows.shape[0]):
            # mask_map = np.zeros((H, W), np.bool_)
            # mask_map[rows[i,:].cpu().numpy(), cols[i,:].cpu().numpy()] = True
            # idx = np.where(mask_map)
            # pts_tgt = np.array(list(zip(idx[1], idx[0]))) # r,c -> (x,y)
            pts_tgt = np.array(list(zip(cols[i,:].cpu().numpy(),
                                        rows[i,:].cpu().numpy())))

            # ========== Test ============
            # pts_tgt = np.array([(100, 200), (200, 300), (300, 400), (400, 500), (400, 100), (400, 200)]).astype(np.float32)
            pts_ipt, status = self.compute_corresponding_point(pts_tgt, self.img_target.astype(np.uint8), self.img_input.astype(np.uint8))
            dep_tgt, valid  = self.compute_depth_value(pts_tgt, pts_ipt, status)
            # ============================

            depth[i] = dep_tgt.flatten()
            mask[i] = valid.flatten()
        return depth, mask
        