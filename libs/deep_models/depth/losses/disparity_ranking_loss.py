import torch
from torch import nn

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

class DisparityRankingLoss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8, sigma=0.15, alpha=0., min_samples=2000):
        super(DisparityRankingLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth
        self.sigma = sigma
        self.alpha = alpha
        self.min_samples = min_samples
    
    def disparityGuidedSamping(self, pred, depth):
        def select_point_pairs(near, far, min_samples, half_min_samples):
            tmp = near[0:min_samples*2:2][:min_samples]
            val = tmp[0:min_samples:2][:half_min_samples]
            inval = tmp[1:min_samples:2][:half_min_samples]

            tmp = far[0:min_samples*2:2][:min_samples]
            valid = tmp[0:min_samples:2][:half_min_samples]
            invalid = tmp[1:min_samples:2][:half_min_samples]
            val = torch.cat((val, valid), dim=0)
            inval = torch.cat((inval, invalid), dim=0)
            
            valid = near[1:min_samples*2:2][:min_samples]
            invalid = far[1:min_samples*2:2][:min_samples]
            val = torch.cat((val, valid), dim=0)
            inval = torch.cat((inval, invalid), dim=0)

            return val, inval

        depth_mask = depth > 0
        # thre computing
        # thre = depth[depth_mask].mean()
        thre = torch.quantile(depth[depth_mask], 0.75)


        mask_A = torch.logical_and(depth <= thre, depth > 0)
        mask_B = depth > thre

        # mask visualization
        import numpy as np
        from matplotlib import pyplot as plt
        # if NPSAVE:
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/dispairty_sampling_A.npy', mask_A.clone().detach().cpu().squeeze().numpy())
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/dispairty_sampling_B.npy', mask_B.clone().detach().cpu().squeeze().numpy())
        # plt.imshow(mask_A.clone().detach().cpu().squeeze().numpy())

        gt_near = depth[mask_A]
        gt_far = depth[mask_B]
        pred_near = pred[mask_A]
        pred_far = pred[mask_B]

        min_samples = min(gt_near.shape[0], gt_far.shape[0]) / 2

        gt_valid, gt_invalid = select_point_pairs(
            torch.sort(gt_near, descending=False)[0], torch.sort(gt_far, descending=True)[0], int(min_samples), int(min_samples/2))
        pred_valid, pred_invalid = select_point_pairs(
            torch.sort(pred_near, descending=False)[0], torch.sort(pred_far, descending=True)[0], int(min_samples), int(min_samples/2))

        # za_gt = (gt_valid - depth[depth_mask].min()) / (depth[depth_mask].max() - depth[depth_mask].min())
        # zb_gt = (gt_invalid- depth[depth_mask].min()) / (depth[depth_mask].max() - depth[depth_mask].min()) 
        za_gt = gt_valid
        zb_gt = gt_invalid

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= (1 + self.sigma)
        mask2 = flag2 > (1 + self.sigma)
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred_valid, pred_invalid, target
    
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

        # disparity-guided sampling
        pred_val, pred_inval, target = self.disparityGuidedSamping(pred_depth, gt_depth)

        loss_mask = self.cal_ranking_loss(pred_val, pred_inval, target)
        loss = loss + loss_mask
        return loss.float()


