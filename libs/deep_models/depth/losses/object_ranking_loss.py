import torch
from torch import nn

from matplotlib import pyplot as plt

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

class ObjectRankingLoss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8, sigma=0.15, alpha=0., min_samples=2000):
        super(ObjectRankingLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth
        self.sigma = sigma
        self.alpha = alpha
        self.min_samples = min_samples
    
    def objectGuidedSampling(self, pred, depth, seg_masks):
        B, C, H, W = depth.shape
        gt_inval, gt_val, pred_inval, pred_val = torch.empty(0).to(device), torch.empty(0).to(device), torch.empty(0).to(device), torch.empty(0).to(device)
        mask_tmp = torch.empty(0).to(device)
        for bs in range(B):
            depth_mask = depth[bs, :, :, :] > 0
            for instance in torch.unique(seg_masks[bs, :, :, :]):
                invalid_mask = seg_masks[bs, :, :, :] == instance
                invalid_mask = torch.logical_and(invalid_mask, depth_mask)
                # 如果seg mask上无特征点，则跳过；若instance从实例1开始，0表示该region无实例
                if not invalid_mask.sum() or instance == 0: 
                    continue
                valid_mask = ~invalid_mask
                valid_mask = torch.logical_and(valid_mask, depth_mask)

                gt_invalid = depth[bs, :, :, :]
                pred_invalid = pred[bs, :, :, :]
                # select the area which belongs to invalid/occlusion
                # mask_invalid = invalid_mask[bs, :, :, :]
                gt_invalid = gt_invalid[invalid_mask]
                pred_invalid = pred_invalid[invalid_mask]

                gt_valid = depth[bs, :, :, :]
                pre_valid = pred[bs, :, :, :]
                # select the area which belongs to valid/reliable
                # mask_valid = valid_mask[bs, :, :, :]
                gt_valid = gt_valid[valid_mask]
                pre_valid = pre_valid[valid_mask]

                # generate the sample index. index range -> (0, len(gt_valid)). The amount -> gt_invalid.size()
                # 参考SC-DepthV3 mask中能用的点都用了
                idx = torch.randint(0, len(gt_valid), gt_invalid.size())
                gt_valid = gt_valid[idx]
                pre_valid = pre_valid[idx]

                # if instance == 1:
                #     gt_inval, gt_val, pred_inval, pred_val = gt_invalid, gt_valid, pred_invalid, pre_valid
                #     continue
                gt_inval = torch.cat((gt_inval, gt_invalid), dim=0)
                gt_val = torch.cat((gt_val, gt_valid), dim=0)
                pred_inval = torch.cat((pred_inval, pred_invalid), dim=0)
                pred_val = torch.cat((pred_val, pre_valid), dim=0)

                # mask visualization
                import numpy as np
                from matplotlib import pyplot as plt
                mask_tmp = torch.cat((mask_tmp, valid_mask*instance), dim=0)

            # save random sampling
            # if NPSAVE:
            #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/object_sampling.npy', mask_tmp.clone().detach().cpu().numpy())

        za_gt = gt_inval
        zb_gt = gt_val

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= (1 + self.sigma)
        mask2 = flag2 > (1 + self.sigma)
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred_inval, pred_val, target

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

        # squared_loss = torch.mean(pred_depth[target == 0] ** 2)

        if torch.isnan(log_loss):
            return torch.DoubleTensor([0.]).cuda()
        
        # if torch.isnan(squared_loss):
        #     return log_loss
        
        return log_loss# + squared_loss


    def forward(self, pred_depth, gt_depth, seg_masks):
        loss = torch.DoubleTensor([0.]).cuda()

        pred_val, pred_inval, target = self.objectGuidedSampling(pred_depth, gt_depth, seg_masks)

        loss_mask = self.cal_ranking_loss(pred_val, pred_inval, target)
        loss = loss + loss_mask #/ (len(torch.unique(seg_masks)))
        return loss.float()


