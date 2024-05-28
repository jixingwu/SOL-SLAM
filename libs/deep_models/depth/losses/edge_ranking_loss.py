import torch
from torch import nn
import torch.nn.functional as F

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def ind2sub(idx, cols):
    r = torch.div(idx, cols, rounding_mode='floor')  # idx // cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

class EdgeRankingLoss(nn.Module):
    def __init__(self, 
                 sample_ratio=0.1, 
                 filter_depth=1e-8, 
                 sigma=0.15, 
                 alpha=1., 
                 min_samples=2000,
                 mask_value = 1e-8):
        super(EdgeRankingLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth
        self.sigma = sigma
        self.alpha = alpha
        self.min_samples = min_samples
        self.mask_value = mask_value
    
    def edgeGuidedSampling(self, inputs, targets, edges_img, thetas_img, masks, h, w, gc):
        # find edges
        # A.ge(B)逐元素对比是否A大于B, i.e. A > B
        # gather(): 沿着由dim指定的轴收集数值
        edges_max = edges_img.max()
        edges_min = edges_img.min()
        edges_mask = edges_img.ge(edges_max * 0.1)
        edges_loc = edges_mask.nonzero(as_tuple=False)

        thetas_edge = torch.masked_select(thetas_img, edges_mask)
        minlen = thetas_edge.size()[0]

        # find anchor points (i.e, edge points)
        sample_num = int(minlen / 10)
        index_anchors = torch.randint(
            0, minlen, (sample_num,), dtype=torch.long).to(device)
        theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
        # 把采样点从id转成 row，col
        row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
        # compute the coordinates of 4-points,  distances are from [2, 30]
        distance_matrix = torch.randint(3, 20, (4, sample_num)).to(device)
        pos_or_neg = torch.ones(4, sample_num).to(device)
        pos_or_neg[:2, :] = -pos_or_neg[:2, :]
        distance_matrix = distance_matrix.float() * pos_or_neg
        col = (
            col_anchors.unsqueeze(0).expand(4, sample_num).long()
            + torch.round(
                distance_matrix.double() * torch.abs(torch.cos(theta_anchors)).unsqueeze(0)
            ).long()
        )
        row = (
            row_anchors.unsqueeze(0).expand(4, sample_num).long()
            + torch.round(
                distance_matrix.double() * torch.abs(torch.sin(theta_anchors)).unsqueeze(0)
            ).long()
        )

        # constrain 0=<c<=w, 0<=r<=h
        # Note: index should minus 1
        col[col < 0] = 0
        col[col > w - 1] = w - 1
        row[row < 0] = 0
        row[row > h - 1] = h - 1

        # mask map for visualization
        import numpy as np
        from matplotlib import pyplot as plt
        # if NPSAVE:
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/edge_sampling_row.npy', row.cpu().numpy())
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/edge_sampling_col.npy', col.cpu().numpy())
        #     plt.figure(1)
        #     mask_map_A = np.zeros((h, w), np.bool_)
        #     mask_map_A[row[0,:].cpu().numpy(), col[0,:].cpu().numpy()] = True
        #     plt.subplot(2,2,1)
        #     plt.imshow(mask_map_A)
        #     mask_map_A[row[1,:].cpu().numpy(), col[1,:].cpu().numpy()] = True
        #     plt.subplot(2,2,2)
        #     plt.imshow(mask_map_A)
        #     mask_map_A[row[2, :].cpu().numpy(), col[2, :].cpu().numpy()] = True
        #     plt.subplot(2,2,3)
        #     plt.imshow(mask_map_A)
        #     mask_map_A[row[3, :].cpu().numpy(), col[3, :].cpu().numpy()] = True
        #     plt.subplot(2,2,4)
        #     plt.imshow(mask_map_A)


        # geometry computing; targets.shape[1,307200]
        depth, mask_tmp = gc.compute_point_depth_for_edge(row, col)
        depth = torch.from_numpy(depth).to(device)
        mask_tmp = torch.from_numpy(mask_tmp).to(device)
        mask = torch.all(mask_tmp, dim=0, keepdim=True).flatten() #(N,)

        # a-b, b-c, c-d
        a = sub2ind(row[0, :], col[0, :], w)[mask]
        b = sub2ind(row[1, :], col[1, :], w)[mask]
        c = sub2ind(row[2, :], col[2, :], w)[mask]
        d = sub2ind(row[3, :], col[3, :], w)[mask]
        A = torch.cat((a, b, c), 0)
        B = torch.cat((b, c, d), 0)

        inputs_A = inputs[:, A]
        inputs_B = inputs[:, B]
        # targets_A = targets[:, A]
        # targets_B = targets[:, B]
        # cat：拼接张量，stack：堆叠张量; cat增加的是维数，stack增加的是维度
        targets_A = torch.cat((depth[0][mask].flatten(),
                               depth[1][mask].flatten(),
                               depth[2][mask].flatten()), dim=0).unsqueeze(0)
        targets_B = torch.cat((depth[1][mask].flatten(),
                               depth[2][mask].flatten(),
                               depth[3][mask].flatten()), dim=0).unsqueeze(0)

        masks_A = torch.gather(masks, 0, A.long())
        masks_B = torch.gather(masks, 0, B.long())

        return (
            inputs_A,
            inputs_B,
            targets_A,
            targets_B,
            masks_A,
            masks_B,
        )


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


    def forward(self, inputs, targets, img_target, gc):
        """
        inputs: predicted depth
        targets: sparse depth treated as ground truth
        """
        # images = torch.from_numpy(np.transpose(gc.img_target, (2,0,1))).float()/255
        masks = targets > self.mask_value
        n,c,h,w = targets.size()
        images = F.interpolate(img_target, (h,w), mode='bilinear', align_corners=False)
        edges_img, thetas_img = self.getEdge(images)

        # mask visualization
        import numpy as np
        from matplotlib import pyplot as plt
        # plt.imshow(edges_mask.clone().detach().cpu().squeeze().numpy())
        # if NPSAVE:
        #     np.save('/media/jixingwu/medisk1/onlineSLAM_output/TUM/for_visual/edges_img.npy',
        #         edges_img.clone().detach().cpu().squeeze().numpy())

        inputs = inputs.contiguous().view(n, c, -1).double()
        targets = targets.contiguous().view(n, c, -1).double()
        masks = masks.contiguous().view(n, -1).double()
        edges_img = edges_img.contiguous().view(n, -1).double()
        thetas_img = thetas_img.contiguous().view(n, -1).double()

        # edge-guided sampling
        loss = torch.DoubleTensor([0.]).cuda()
        for i in range(n):
            (
                inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
            ) = self.edgeGuidedSampling(
                inputs[i,:], targets[i,:], edges_img[i], thetas_img[i], masks[i,:], h, w, gc
            )

            consistency_mask = masks_A * masks_B

            # GT ordinal relationship
            # lt: 小于，gt: 相等
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask
            
            loss = loss + equal_loss.mean() + unequal_loss.mean()

        return loss[0].float() / n


