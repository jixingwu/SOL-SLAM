''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: DeepModel initializes different deep networks and provide forward interfaces.
'''

import numpy as np
import os
import PIL.Image as pil
import torch
import torch.optim as optim
from torchvision import transforms

from .flow.lite_flow_net.lite_flow import LiteFlow
from libs.general.utils import mkdir_if_not_exists

from .depth.trainer_on import TrainerOn
from .depth.config import Paraser, get_training_size
import libs.deep_models.depth.datasets.custom_transforms as custom_transforms
from path import Path
from .depth.visualization import *
from imageio.v2 import imread, imwrite

class DeepModel():
    def __init__(self, cfg):
        self.cfg = cfg
        self.finetune_cfg = self.cfg.online_finetune
        self.device = torch.device('cuda')


    def initialize_models(self):
        """ initialize deep models """
        ''' optical flow '''
        if self.cfg.off_flownet:
            print("==> Load flow results from {}".format(self.cfg.directory.flow_dir))
        else:
            self.flow = self.initialize_deep_flow_model()

        ''' single-view depth '''
        if self.cfg.depth.deep_depth.pretrained_model is not None:
            self.depth = self.initialize_deep_depth_model()
        else:
            assert False, "No precomputed depths nor pretrained depth model"

    def initialize_deep_flow_model(self):
        if self.cfg.deep_flow.network == 'liteflow':
            flow_net = LiteFlow(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.flow.enable
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight,
                    finetune=enable_finetune,
                    )
        else:
            assert False, "Invalid flow network [{}] is provided.".format(
                                self.cfg.deep_flow.network
                                )
        return flow_net

    def initialize_deep_depth_model(self):
        if self.cfg.depth.deep_depth.network == 'sc_depthv3':
            hparams = Paraser()
            hparams.dataset_name = self.cfg.dataset
            self.output_dir = Path(self.cfg.directory.result_dir) / 'model_{}_{}_r{}'.format(hparams.model_version, self.cfg.online_learning.mode, self.cfg.online_learning.lora_r)
            self.output_dir.makedirs_p()
            (self.output_dir / 'vis').makedirs_p()
            (self.output_dir / 'depth').makedirs_p()

            # training size
            training_size = get_training_size(hparams.dataset_name)
            self.inference_transform = custom_transforms.Compose([
                custom_transforms.RescaleTo(training_size),
                custom_transforms.ArrayToTensor(),
                custom_transforms.Normalize()]
            )
            self.train_transform = custom_transforms.Compose([
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.RandomScaleCrop(),
                custom_transforms.RescaleTo(training_size),
                custom_transforms.ArrayToTensor(),
                custom_transforms.Normalize()]
            )
            self.valid_transform = custom_transforms.Compose([
                custom_transforms.RescaleTo(training_size),
                custom_transforms.ArrayToTensor(),
                custom_transforms.Normalize()]
            )
            enable_learning = self.cfg.online_learning.enable
            depth_net = TrainerOn(self.cfg, self.cfg.depth.deep_depth.pretrained_model, hparams, enable_learning)
        else:
            assert False, "Invalid depth network [{}] is provided.".format(
                                self.cfg.depth.deep_depth.network
                                )
        return depth_net

    def forward_flow(self, in_cur_data, in_ref_data, forward_backward):
        """Optical flow network forward interface, a forward inference.

        Args:
            in_cur_data (dict): current data
            in_ref_data (dict): reference data
            forward_backward (bool): use forward-backward consistency if True
        
        Returns:
            flows (dict): predicted flow data. flows[(id1, id2)] is flows from id1 to id2.

                - **flows(id1, id2)** (array, 2xHxW): flows from id1 to id2
                - **flows(id2, id1)** (array, 2xHxW): flows from id2 to id1
                - **flows(id1, id2, 'diff)** (array, 1xHxW): flow difference of id1
        """
        # Preprocess image
        cur_imgs = np.transpose((in_cur_data['img'])/255, (2, 0, 1))
        ref_imgs = np.transpose((in_ref_data['img'])/255, (2, 0, 1))
        cur_imgs = torch.from_numpy(cur_imgs).unsqueeze(0).float().cuda()
        ref_imgs = torch.from_numpy(ref_imgs).unsqueeze(0).float().cuda()

        # Forward pass
        flows = {}

        # Flow inference
        batch_flows = self.flow.inference_flow(
                                img1=ref_imgs,
                                img2=cur_imgs,
                                forward_backward=forward_backward,
                                dataset=self.cfg.dataset)
        
        # Save flows at current view
        src_id = in_ref_data['id']
        tgt_id = in_cur_data['id']
        flows[(src_id, tgt_id)] = batch_flows['forward'].detach().cpu().numpy()[0]
        if forward_backward:
            flows[(tgt_id, src_id)] = batch_flows['backward'].detach().cpu().numpy()[0]
            flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'].detach().cpu().numpy()[0]
        return flows

    def forward_depth(self, img, timestamp):
        """Depth network forward interface, a forward inference.

        Args:
            imgs (list): list of images, each element is a [HxWx3] array

        Returns:
            depth (array, [HxW]): depth map of imgs[0]
        """
        # Preprocess
        tensor_img = self.inference_transform([img.astype(np.float32)])[0][0].unsqueeze(0).cuda()
        # Inference
        with torch.no_grad():
            pred_depth = self.depth.model(tensor_img)
            depth = pred_depth[0, 0].clone().detach().cpu().numpy()

        vis = visualize_depth(depth).permute(1, 2, 0).numpy() * 255
        # imwrite(self.output_dir / 'vis/{:0>10}.jpg'.format(timestamp), vis.astype(np.uint8))
        # np.save(self.output_dir / 'depth/{:0>10}.npy'.format(timestamp), depth)

        imwrite(self.output_dir / 'vis/{:.6f}.jpg'.format(timestamp), vis.astype(np.uint8))
        np.save(self.output_dir / 'depth/{:.6f}.npy'.format(timestamp), depth)

        return depth

    def finetune(self, img1, img2, pose, K, inv_K):
        """Finetuning deep models

        Args:
            img1 (array, [HxWx3]): image 1 (reference)
            img2 (array, [HxWx3]): image 2 (current)
            pose (array, [4x4]): relative pose from view-2 to view-1 (from DF-VO)
            K (array, [3x3]): camera intrinsics
            inv_K (array, [3x3]): inverse camera intrinsics
        """
        # preprocess data
        # images
        img1 = np.transpose((img1)/255, (2, 0, 1))
        img2 = np.transpose((img2)/255, (2, 0, 1))
        img1 = torch.from_numpy(img1).unsqueeze(0).float().cuda()
        img2 = torch.from_numpy(img2).unsqueeze(0).float().cuda()

        # camera intrinsics
        K44 = np.eye(4)
        K44[:3, :3] = K.copy()
        K = torch.from_numpy(K44).unsqueeze(0).float().cuda()
        K44[:3, :3] = inv_K.copy()
        inv_K = torch.from_numpy(K44).unsqueeze(0).float().cuda()

        # pose
        if self.finetune_cfg.depth.pose_src == 'DF-VO':
            pose = torch.from_numpy(pose).unsqueeze(0).float().cuda()
            pose[:, :3, 3] /= 5.4
        elif self.finetune_cfg.depth.pose_src == 'deep_pose':
            pose = self.pose.pred_pose
        elif self.finetune_cfg.depth.pose_src == 'DF-VO2':
            deep_pose_scale = torch.norm(self.pose.pred_pose[:, :3, 3].clone())
            pose = torch.from_numpy(pose).unsqueeze(0).float().cuda()
            pose[:, :3, 3] /= torch.norm(pose[:, :3, 3])
            pose[:, :3, 3] *= deep_pose_scale
        
        if self.finetune_cfg.num_frames is None or self.img_cnt < self.finetune_cfg.num_frames:
            ''' data preparation '''
            losses = {'loss': 0}
            inputs = {
                ('color', 0, 0): img1,
                ('color', 1, 0): img2,
                ('K', 0): K,
                ('inv_K', 0): inv_K,
            }
            outputs = {}

            ''' loss computation '''
            # flow
            if self.finetune_cfg.flow.enable:
                assert self.cfg.deep_flow.forward_backward, "forward-backward option has to be True for finetuning"
                for s in self.flow.flow_scales:
                    outputs.update(
                        {
                            ('flow', 0, 1, s):  self.flow.forward_flow[s],
                            ('flow', 1, 0, s):  self.flow.backward_flow[s],
                            ('flow_diff', 0, 1, s):  self.flow.flow_diff[s]
                        }
                    )

                losses.update(self.flow.train(inputs, outputs))
                losses["loss"] += losses["flow_loss"]
            
            # depth and pose
            if self.finetune_cfg.depth.enable:
                # add predicted depths
                for s in self.depth.depth_scales:
                    outputs.update(
                        {
                            ('depth', 1, s): self.depth.pred_depths[s][:1],
                            ('disp', 1, s): self.depth.pred_disps[s][:1],
                            ('depth', 0, s): self.depth.pred_depths[s][1:],
                            ('disp', 0, s): self.depth.pred_disps[s][1:]
                        }
                    )

                # add predicted poses
                outputs.update(
                    {
                        ('pose_T', 1, 0): pose
                    }
                )
                
                losses.update(self.depth.train(inputs, outputs))
                losses["loss"] += losses["reproj_sm_loss"]
                if self.depth.depth_consistency != 0:
                    losses["loss"] += losses["depth_consistency_loss"]
            
            ''' backward '''
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            self.img_cnt += 1

        else:
            # reset depth model to eval mode
            if self.finetune_cfg.depth.enable:
                self.depth.model.eval()

    def online_learning(self):
        """Online learning for deep models
        """
    def save_model(self):
        """Save deep models
        """
        save_folder = os.path.join(self.cfg.directory.result_dir, "deep_models", self.cfg.seq)
        mkdir_if_not_exists(save_folder)

        # Save Flow model
        model_name = "flow"
        model = self.flow.model
        ckpt_path = os.path.join(save_folder, "{}.pth".format(model_name))
        torch.save(model.state_dict(), ckpt_path)
