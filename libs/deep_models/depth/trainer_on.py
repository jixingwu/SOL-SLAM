import os
import torch
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import libs.deep_models.depth.losses.loss_functions as LossF
from kornia.geometry.depth import depth_to_normals

from libs.deep_models.depth.odometry.calibration import Calibration

from libs.deep_models.depth.SC_Depth import SC_Depth
from libs.deep_models.depth.utils.camParams import *

import matplotlib.pyplot as plt

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

def to_cuda(item):
    return item.float().unsqueeze(0).cuda()

class TrainerOn:
    def __init__(self, cfg, ckpt_path, hparams, online_enable):
        self.cfg = cfg
        self.hparams = hparams
        self.hparams.lora_r = self.cfg.online_learning.lora_r
        self.hparams.lora_r2 = self.cfg.online_learning.lora_r2
        K, _ = get_intrinsics_params(self.hparams.dataset_name)
        # 初始化模型
        self.system = SC_Depth(self.hparams, 
                               self.cfg.online_learning.lora_r, 
                               self.cfg.online_learning.lora_r2)
        self.system = self.system.load_from_checkpoint(ckpt_path, 
                                                       lora_r=self.cfg.online_learning.lora_r, lora_r2=self.cfg.online_learning.lora_r2, strict=False)
        self.model = self.system.depth_net.cuda().eval()

        self.step = 0
        self.batch_size = self.hparams.batch_size
        self.online_enable = online_enable
        self.abs_rel_errs = 1.
        self.intrinsics = to_cuda(torch.from_numpy(K))
        self.calib = Calibration(self.hparams.dataset_name)
        self.has_printed = False

        # optimizer
        optim_params = [
            {'params': self.model.parameters(), 'lr': self.hparams.lr}
            ]
        self.model_optimizer = optim.Adam(optim_params)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 100, 0.1)

        if self.online_enable:
            self.set_train()
        else:
            self.set_eval()
        
        print(f" Total Pramerters: {sum(p.numel() for p in self.model.parameters())/1000/1000} MB")
        print(f" Trainable Pramerters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1000/1000} MB")

        self.writers = {}
        for mode in ["train", "val"]:
           self.writers[mode] = SummaryWriter(os.path.join(self.hparams.log_path, mode))

    def set_train(self):
        mode = self.cfg.online_learning.mode #[refiner, finetune, bn]

        if not self.has_printed:
            print('Set model to training mode: {}, lora_r: {}'.format(mode, self.cfg.online_learning.lora_r))
            self.has_printed = True

        self.model.train()

        if mode == 'refiner':
            self.model.requires_grad_(False)
            # setting bn
            for _, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)

            for name, param in self.model.named_parameters():
                # param.requires_grad = False
                if 'lora_' in name:
                    param.requires_grad = True

        elif mode == 'finetune':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                if 'lora_' in name:
                    param.requires_grad = False
        elif mode == 'bn':
            self.model.requires_grad_(False)
            for _, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                
    def set_eval(self):
        self.model.eval()
    
    def run_epoch(self, inputs):
        # inputs: (list) ref_imgs, tgt_img, poses, tgt_sparse_depth, tgt_seg_mask, ref_seg_masks， tgt_seg_id

        """===== prepare train data format ====="""
        tgt_img, tgt_depth_spa, tgt_seg = torch.empty(0).cuda(), torch.empty(0).cuda(),torch.empty(0).cuda()

        tgt_seg_ids = []
        ref_imgs = []
        ref_segs = []
        poses = []
        poses_inv = []

        # concat and convert
        for input in inputs:
            # tgt_img 
            tgt_img = torch.cat((tgt_img, to_cuda(input[1])), dim=0)                    
            # tgt_sparse_depth and seg_masks
            tgt_depth_spa = torch.cat((tgt_depth_spa, to_cuda(input[3])))           
            tgt_seg = torch.cat((tgt_seg, to_cuda(input[4])))
            tgt_seg_ids.append(input[6])
        
        # 是因为tgt_img是一个batch的数据作为整体，与ref_imgs计算是与list中每个成员计算，因此每个成员都是一个batch的数据
        for item in range(len(inputs[0][0])):
            ref_img, pose, pose_inv, ref_seg = torch.empty(0).cuda(), torch.empty(0).cuda(), torch.empty(0).cuda(), torch.empty(0).cuda()

            # ref_imgs 
            for input in inputs:                                  
                ref_img = torch.cat((ref_img, to_cuda(input[0][item])), dim=0)
            ref_imgs.append(ref_img)

            # ref segs
            for input in inputs:                                  
                ref_seg = torch.cat((ref_seg, to_cuda(input[5][item])), dim=0)
            ref_segs.append(ref_seg)

            # poses and inverse poses
            for input in inputs:
                pose = torch.cat((pose, to_cuda(input[2][item])), dim=0)                    
                pose_inv = torch.cat((pose_inv, torch.inverse(to_cuda(input[2][item]))), dim=0) 
            poses.append(pose)
            poses_inv.append(pose_inv)


        """===== Network forward ====="""
        tgt_depth = self.model(tgt_img) # [B, C, H, W]
        ref_depths = [self.model(im) for im in ref_imgs]

        tgt_seg = F.interpolate(tgt_seg, tgt_depth.shape[2:], mode='nearest')
        ref_segs = [F.interpolate(seg, tgt_depth.shape[2:], mode='nearest') for seg in ref_segs]
        tgt_sparse_extend = F.interpolate(tgt_depth_spa, tgt_depth.shape[2:], mode='bilinear', align_corners=False)

        """===== loss computing ======"""
        loss_total = torch.DoubleTensor([0.]).cuda()
        w1, w2, w3, w4 = 0., 0., 0.1, 0.1
        DCR = True

        if DCR:
            loss_1, loss_2, valid_mask = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth,       ref_depths, self.intrinsics, poses, poses_inv, self.hparams, tgt_seg, ref_segs, tgt_seg_ids, True)
        else:
            loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, self.intrinsics, poses, poses_inv, self.hparams, None, None, False)
        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)

        mask = torch.logical_and(tgt_sparse_extend > 0, valid_mask)
        loss_4 = (tgt_depth[mask] - tgt_sparse_extend[mask]).abs().mean()

        loss_total = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4
        
        self.model_optimizer.zero_grad()
        loss_total.backward()
        self.model_optimizer.step()
        self.model_lr_scheduler.step()

        """===== visualize loss curves"""
        # write to tensorboard
        writer = self.writers["train"]
        writer.add_scalars("L", {
            "total_loss":        loss_total.item(),
            "photo_loss":        loss_1.item(),
            "geometry_loss":     loss_2.item(),
            "smooth_loss":       loss_3.item(),
            "disparity_loss":    loss_4.item()
        }, self.step)
        
        self.step += 1
        # 释放不再需要的 GPU 内存
        torch.cuda.empty_cache()
    
    def training_step(self, tgt_depth, batch, gc):

        ref_imgs, tgt_img, poses, tgt_sparse_depth, seg_masks = batch

        # for bs in self.hparams.batch_size:
        ref_depths = [self.model(im) for im in ref_imgs]
        
        poses_inv = [torch.inverse(pose) for pose in poses]
        
        _, C, H, W = tgt_depth.shape
        seg_masks_inter = F.interpolate(seg_masks, tgt_depth.shape[2:], mode='bilinear', align_corners=False)
        _, _, H1, W1 = tgt_sparse_depth.shape
        tgt_depth_extend = F.interpolate(tgt_depth, (H1, W1), mode='bilinear', align_corners=False)
        
        # compute normal
        # tgt_pseudo_normal = depth_to_normals(tgt_pseudo_depth, intrinsics)
        # tgt_normal = depth_to_normals(tgt_depth, intrinsics)
        # tgt_sparse_normal = depth_to_normals(tgt_sparse_detph, intrinsics)
        # tgt_normal_tmp = depth_to_normals(tgt_depth_tmp, intrinsics)  

        loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                       self.intrinsics, poses, poses_inv, self.hparams, seg_masks_inter)
        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)
        
        # ranking loss：增强mask之间区分
        # loss_4 = LossF.ranking_loss(tgt_depth_extend, tgt_sparse_depth, seg_masks)
        # random sampling
        loss_41 = LossF.random_ranking_loss(tgt_depth_extend, tgt_sparse_depth)
        # object sampling
        loss_42 = LossF.object_ranking_loss(tgt_depth_extend, tgt_sparse_depth, seg_masks)
        # disparity sampling
        loss_43 = LossF.disparity_ranking_loss(tgt_depth_extend, tgt_sparse_depth)
        #edge sampling
        loss_44 = LossF.edge_ranking_loss(tgt_depth_extend, tgt_sparse_depth, tgt_img, gc)
        
        # abs rel loss
        sparse_mask = tgt_sparse_depth > 0
        ratio = torch.median(tgt_sparse_depth[sparse_mask]) / torch.median(tgt_depth_extend[sparse_mask])
        loss_5 = torch.abs((ratio * tgt_depth_extend[sparse_mask] - tgt_sparse_depth[sparse_mask]) / (ratio * tgt_depth_extend[sparse_mask]))

        return [loss_1, loss_2, loss_3, loss_41, loss_42, loss_43, loss_44, loss_5.mean()]
    
    
    def numpy2Tensor(self, batch):
        
        # batch = [ref_imgs, cur_imgs, ref_pose, pse_depth, spa_depth, seg_mask, K]
        tensor = []
        tensor.append([to_cuda(elem) for elem in batch[0]])                   # ref_imgs
        tensor.append(to_cuda(batch[1]))                                      # tgt_img
        tensor.append([to_cuda(torch.from_numpy(elem)) for elem in batch[2]]) # poses
        tensor.append(to_cuda(torch.from_numpy(batch[3]).unsqueeze(0)))                    # tgt_sparse_depth
        tensor.append(to_cuda(torch.from_numpy(batch[4]).unsqueeze(0)))       # seg_masks
        tensor.append(to_cuda(torch.from_numpy(batch[5])))                    # intrinsics
        return tensor
        
    def val(self, val_dict):
        self.set_eval()
        errors = []
        
        with torch.no_grad():
            for data, gt_depth in zip(val_dict['imgs'], val_dict['deps']):
                input_color = data.cuda()
                h, w  = gt_depth.shape
                pred_depth = self.model(input_color).detach().cpu().float()
                pred_depth = F.interpolate(pred_depth, gt_depth.shape, mode='bilinear', align_corners=False)
                pred_depth = pred_depth.numpy()[0,0]
                
                min_depth = 0.1
                if self.hparams.dataset_name == 'kitti_odom':
                    max_depth = 80.0
                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
                    crop = np.array([0.40810811 * h,  0.99189189 * h,
                                     0.03594771 * w, 0.96405229 * w]).astype(np.int32)
                    crop_mask = np.zeros_like(mask)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                if self.hparams.dataset_name in ['tum-2', 'tum-3']:
                    max_depth = 80.0
                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

                pred_depth[pred_depth < min_depth] = min_depth
                pred_depth[pred_depth > max_depth] = max_depth
                
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
                ratio = np.median(gt_depth) / np.median(pred_depth)
                pred_depth *= ratio
                    
                errs = LossF.compute_depth_errors(gt_depth, pred_depth)
                errors.append(errs)
                
        mean_errors = np.array(errors).mean(0)
        
        # TODO: save model parameters
        # if mean_errors[0] < self.abs_rel_errs:
        #     self.abs_rel_errs = mean_errors[0]
        #     save_path = '/media/jixingwu/medisk1/Online-SLAM/ckpts/ddad2kittiOnline'
        #     torch.save(self.model.state_dict(), os.path.join(save_path, 
        #                 'epoch={}-val_loss={:.4f}.ckpt'.format(self.step, self.abs_rel_errs)))
        
        # write to tensorboard
        writer = self.writers["val"]
        for l, v in zip(["abs_rel", "sq_rel", "rmse", "rmse_log", "lg10", "a1", "a2", "a3"],
                mean_errors.tolist()):
            if l in ["abs_rel", "sq_rel", "rmse", "rmse_log", "lg10"]:
                writer.add_scalar("error/{}".format(l), v, self.step)
            else:
                writer.add_scalar("acc/{}".format(l), v, self.step)
        self.set_train()