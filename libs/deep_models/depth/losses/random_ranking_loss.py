import torch
from torch import nn


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

class RandomRankingLoss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8, sigma=0.15, alpha=0., min_samples=2000):
        super(RandomRankingLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth
        self.sigma = sigma
        self.alpha = alpha
        self.min_samples = min_samples

    def randomSampling(self, pred, depth):
        # .nelement()统计张量个数，
        # .randperm(n)将0~n-1随机打乱后获得数字序列
        # .view(-1) 将张量重构成一维向量
        # .repeat(B,1,1,1)在B维度上重复B次
        B, C, H, W = depth.shape
        depth_mask = depth > 0
        depth_tmp = depth.clone()
        depth = depth[depth_mask]
        pred = pred[depth_mask]
        mask_A = torch.rand(depth.size()).to(device)
        mask_A[mask_A >= (1 - self.sample_ratio)] = 1
        mask_A[mask_A < (1 - self.sample_ratio)] = 0
        idx = torch.randperm(mask_A.nelement())
        mask_B = mask_A.view(-1)[idx].view(mask_A.size())
        mask_A = mask_A.repeat(B, 1, 1, 1).view(depth.shape) == 1
        mask_B = mask_B.repeat(B, 1, 1, 1).view(depth.shape) == 1
        za_gt = depth[mask_A]
        zb_gt = depth[mask_B]
        mask_ignoreb = zb_gt > self.filter_depth
        mask_ignorea = za_gt > self.filter_depth
        mask_ignore = mask_ignorea | mask_ignoreb
        za_gt = za_gt[mask_ignore]
        zb_gt = zb_gt[mask_ignore]

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= (1 + self.sigma)
        mask2 = flag2 > (1 + self.sigma)
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        # mask map for visualization
        import numpy as np
        from matplotlib import pyplot as plt
        mask_A_tmp = torch.rand(depth_tmp.size()).to(device)
        mask_A_tmp[mask_A_tmp >= (1 - self.sample_ratio)] = 1
        mask_A_tmp[mask_A_tmp < (1 - self.sample_ratio)] = 0
        idx = torch.randperm(mask_A_tmp.nelement())
        mask_B_tmp = mask_A_tmp.view(-1)[idx].view(mask_A_tmp.size())
        mask_A_tmp = mask_A_tmp.repeat(B, 1, 1, 1).view(depth_tmp.shape) == 1
        mask_B_tmp = mask_B_tmp.repeat(B, 1, 1, 1).view(depth_tmp.shape) == 1
        mask_A_tmp = mask_A_tmp * depth_mask
        mask_B_tmp = mask_B_tmp * depth_mask

        mask_map_A = np.zeros((H,W), np.bool_)
        mask_map_A[mask_A_tmp.clone().detach().cpu().squeeze().numpy().astype(np.bool_)] = True
        mask_map_B = np.zeros((H,W), np.bool_)
        mask_map_B[mask_B_tmp.clone().detach().cpu().squeeze().numpy().astype(np.bool_)] = True
        # save random sampling
        # if NPSAVE:
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/feature_point_mask.npy', depth_mask.clone().detach().cpu().squeeze().numpy())
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/random_sampling_A.npy', mask_map_A)
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/random_sampling_B.npy', mask_map_B)

        return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target
        
    def cal_ranking_loss(self, z_A, z_B, target):
        """ 计算l = -1 或 1 或 0
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        pred_depth = z_A - z_B

        log_loss = torch.mean(
            torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))

        squared_loss = torch.mean(pred_depth[target == 0] ** 2)

        if torch.isnan(log_loss):
            return squared_loss
        
        if torch.isnan(squared_loss):
            return log_loss
        
        return log_loss + squared_loss


    def forward(self, pred_depth, gt_depth):
        loss = torch.DoubleTensor([0.]).cuda()

        # random_sampling
        pred_val, pred_inval, target = self.randomSampling(pred_depth, gt_depth)

        loss_mask = self.cal_ranking_loss(pred_val, pred_inval, target)
        loss = loss + loss_mask
        return loss.float()


