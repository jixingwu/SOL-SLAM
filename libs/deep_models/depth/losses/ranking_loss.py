import torch
from torch import nn

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

class Ranking_Loss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8, sigma=0.15, alpha=0., min_samples=2000):
        super(Ranking_Loss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth
        self.sigma = sigma
        self.alpha = alpha
        self.min_samples = min_samples

    def segmentGuidedSampling(self, pred, depth, seg_masks):
        pred_inval, pred_val, gt_inval, gt_val, mask_A, mask_B = None, None, None, None, None, None
        depth_mask = depth > 0

        for seg_mask in seg_masks:

            valid_mask = torch.logical_and(seg_mask, depth_mask)
            invalid_mask = torch.logical_and(~seg_mask.bool(), depth_mask)

            if valid_mask.sum() == 0:
                continue
            
            gt_valid = depth[valid_mask]
            pred_valid = pred[valid_mask]
            mask_a = valid_mask[valid_mask]

            idx_invalid = torch.randint(0, invalid_mask.sum(), (gt_valid.shape[0],)).cuda()
            gt_invalid =  torch.gather(depth[invalid_mask], 0, idx_invalid)
            pred_invalid = torch.gather(pred[invalid_mask], 0, idx_invalid)
            mask_b = torch.gather(invalid_mask[invalid_mask], 0, idx_invalid)

            if gt_inval is None:
                gt_inval, gt_val, pred_inval, pred_val, mask_A, mask_B = gt_invalid, gt_valid, pred_invalid, pred_valid, mask_a, mask_b
                continue

            gt_inval = torch.cat((gt_inval, gt_invalid), dim=0)
            gt_val = torch.cat((gt_val, gt_valid), dim=0)
            pred_inval = torch.cat((pred_inval, pred_invalid), dim=0)
            pred_val = torch.cat((pred_val, pred_valid), dim=0)
            mask_A = torch.cat((mask_A, mask_a), dim=0)
            mask_B = torch.cat((mask_B, mask_b), dim=0)

        if pred_val is None:
            pred_val, pred_inval, gt_val, gt_inval, mask_A, mask_B = self.randomSampling(pred, depth)

        za_gt = (gt_valid - depth[depth_mask].min()) / (depth[depth_mask].max() - depth[depth_mask].min())
        zb_gt = (gt_invalid- depth[depth_mask].min()) / (depth[depth_mask].max() - depth[depth_mask].min()) 

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= (1 + self.sigma)
        mask2 = flag2 > (1 + self.sigma)
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred_valid, pred_invalid, target

    def randomSampling(self, pred, depth):
        # .nelement()统计张量个数，
        # .randperm(n)将0~n-1随机打乱后获得数字序列
        # .view(-1) 将张量重构成一维向量
        # .repeat(B,1,1,1)在B维度上重复B次
        B, C, H, W = depth.shape
        depth_mask = depth > 0
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

        return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target
    
    def objectGuidedSampling(self, pred, depth, seg_masks):
        B, C, H, W = depth.shape
        depth_mask = depth > 0
        gt_inval, gt_val, pred_inval, pred_val = torch.empty(0).to(device), torch.empty(0).to(device), torch.empty(0).to(device), torch.empty(0).to(device)
        for bs in range(B):
            for instance in torch.unique(seg_masks):
                invalid_mask = seg_masks == instance
                invalid_mask = torch.logical_and(invalid_mask, depth_mask)
                # 如果seg mask上无特征点，则跳过；若instance从实例1开始，0表示该region无实例
                if not invalid_mask.sum() or instance == 0: 
                    continue
                valid_mask = ~invalid_mask
                valid_mask = torch.logical_and(valid_mask, depth_mask)

                gt_invalid = depth[bs, :, :, :]
                pred_invalid = pred[bs, :, :, :]
                # select the area which belongs to invalid/occlusion
                mask_invalid = invalid_mask[bs, :, :, :]
                gt_invalid = gt_invalid[mask_invalid]
                pred_invalid = pred_invalid[mask_invalid]

                gt_valid = depth[bs, :, :, :]
                pre_valid = pred[bs, :, :, :]
                # select the area which belongs to valid/reliable
                mask_valid = valid_mask[bs, :, :, :]
                gt_valid = gt_valid[mask_valid]
                pre_valid = pre_valid[mask_valid]

                # generate the sample index. index range -> (0, len(gt_valid)). The amount -> gt_invalid.size()
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

    def edgeGuidedSampling(self, pred, depth, edges):
        edges_img, thetas_img = self.getEdge
    
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
        depth_mean = depth[depth_mask].mean()

        gt_near = depth[torch.logical_and(depth <= depth_mean, depth > 0)]
        gt_far = depth[depth > depth_mean]

        pred_near = pred[torch.logical_and(depth <= depth_mean, depth > 0)]
        pred_far = pred[depth > depth_mean]

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
    
    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas


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


    def forward(self, pred_depth, gt_depth, seg_masks):
        loss = torch.DoubleTensor([0.]).cuda()

        R = True
        O = True
        E = False
        D = True

        pred_val, pred_inval, target = torch.empty(0).to(device), torch.empty(0).to(device), torch.empty(0).to(device)

        # random_sampling
        if R:
            pred_val_, pred_inval_, target_ = self.randomSampling(pred_depth, gt_depth)
            pred_val = torch.cat((pred_val, pred_val_), dim=0)
            pred_inval = torch.cat((pred_inval, pred_inval_), dim=0)
            target = torch.cat((target, target_), dim=0)
        
        # object-guided sampling
        if O:
            pred_val_, pred_inval_, target_ = self.objectGuidedSampling(pred_depth, gt_depth, seg_masks)
            pred_val = torch.cat((pred_val, pred_val_), dim=0)
            pred_inval = torch.cat((pred_inval, pred_inval_), dim=0)
            target = torch.cat((target, target_), dim=0)

        # edge-guided sampling
        if E:
            pred_val_, pred_inval_, target_ = self.edgeGuidedSampling(pred_depth, gt_depth, seg_masks)
            pred_val = torch.cat((pred_val, pred_val_), dim=0)
            pred_inval = torch.cat((pred_inval, pred_inval_), dim=0)
            target = torch.cat((target, target_), dim=0)
        # disparity-guided sampling
        if D:
            pred_val_, pred_inval_, target_ = self.disparityGuidedSamping(pred_depth, gt_depth)
            pred_val = torch.cat((pred_val, pred_val_), dim=0)
            pred_inval = torch.cat((pred_inval, pred_inval_), dim=0)
            target = torch.cat((target, target_), dim=0)

        assert R or O or E or D

        loss_mask = self.cal_ranking_loss(pred_val, pred_inval, target)
        loss = loss + loss_mask
        return loss.float()


