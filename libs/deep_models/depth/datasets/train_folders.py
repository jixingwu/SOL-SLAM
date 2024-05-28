import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
from path import Path
import random

class TrainBatch(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def getitem(self):
        ref_imgs, tgt_img, poses, tgt_sparse_depth, seg_mask, ref_seg_masks, intrinsics, seg_mask_id = self.data

        poses_tmp = [torch.from_numpy(pose) for pose in poses]
        tgt_sparse_depth = torch.from_numpy(tgt_sparse_depth).unsqueeze(0)
        seg_mask = torch.from_numpy(seg_mask).unsqueeze(0)
        ref_seg_masks = [torch.from_numpy(ref_seg_mask).unsqueeze(0) for ref_seg_mask in ref_seg_masks]

        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)

        return ref_imgs, tgt_img, poses_tmp, tgt_sparse_depth, seg_mask, ref_seg_masks, seg_mask_id