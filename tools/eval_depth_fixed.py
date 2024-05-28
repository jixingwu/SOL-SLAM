import argparse
from re import I
import cv2
import numpy as np
import os
from tqdm import tqdm
from path import Path
from imageio import imread
from scipy import sparse
import glob
import copy

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluation scripts")
parser.add_argument("--dataset", required=True, help="kitti or nyu",
                    choices=['nyu', 'bonn', 'tum', 'kitti', 'ddad', 'scannet', 'euroc'], type=str)
parser.add_argument("--pred_depth", required=True,
                    help="predicted depth folders", type=str)
parser.add_argument("--gt_depth", required=True,
                    help="gt depth folders", type=str)
parser.add_argument("--seg_mask", default=None,
                    help="segmentation mask folders", type=str)

######################################################
args = parser.parse_args()


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense())
    return depth


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    mae = np.mean(np.abs(gt - pred))

    return abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3, mae

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    rgb_asso = []
    depth_asso = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            rgb_asso.append(a)
            depth_asso.append(b)
    
    rgb_asso.sort()
    depth_asso.sort()
    return rgb_asso, depth_asso

def align_img_with_pose_and_depth():
    rgb_list = read_file_list('/media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/rgb.txt')
    depth_list = read_file_list('/media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/depth.txt')
    pose_list = read_file_list('/media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/groundtruth.txt')
    
    rgb_d_pose_pair = {}
    for i in rgb_list:
        rgb_d_pose_pair[i] = {}

    # associate rgb-d
    matches = associate(
        first_list=rgb_list,
        second_list=depth_list,
        offset=0,
        max_difference=0.02
    )
    for match in matches:
        rgb_stamp = match[0]
        depth_stamp = match[1]
        rgb_d_pose_pair[rgb_stamp]['depth'] = depth_stamp
    
    # associate rgb-pose
    matches = associate(
        first_list=rgb_list,
        second_list=pose_list,
        offset=0,
        max_difference=0.02
    )
    
    for match in matches:
        rgb_stamp = match[0]
        pose_stamp = match[1]
        rgb_d_pose_pair[rgb_stamp]['pose'] = pose_stamp
    
    # # Clear pairs without pose
    to_del_pair = []
    tmp_rgb_d_pose_pair = copy.deepcopy(rgb_d_pose_pair)
    for rgb_stamp in tmp_rgb_d_pose_pair:
        if rgb_d_pose_pair[rgb_stamp].get('pose', -1) == -1:
            to_del_pair.append(rgb_stamp)
    for rgb_stamp in to_del_pair:
        del(rgb_d_pose_pair[rgb_stamp])
    
    # timestep
    timestep = 1
    to_del_pair = []
    for cnt, rgb_stamp in enumerate(rgb_d_pose_pair):
        if cnt % timestep != 0:
            to_del_pair.append(rgb_stamp)
    for rgb_stamp in to_del_pair:
        del(rgb_d_pose_pair[rgb_stamp])
    
    return rgb_d_pose_pair

class DepthEval():
    def __init__(self):

        self.min_depth = 0.1
        
        if args.dataset == 'kitti':
            self.max_depth = 80.
        elif args.dataset == 'tum':
            self.max_depth = 30.
        elif args.dataset == 'euroc':
            self.max_depth = 10.

    def main(self):
        pred_depths = []

        """ Get results """
        pred_depths = sorted(Path(args.pred_depth).files("*.npy"))  # in *.jpg
        
        self.pred_depth = args.pred_depth
        """ get gt depths """
        if args.dataset == 'tum':
            gt_depths = sorted(Path(args.gt_depth).files("*.png"))  # in *.png
        elif args.dataset == 'kitti':
            gt_depths = sorted(Path(args.gt_depth).files("*.png"))  # in *.png
        elif args.dataset == 'euroc':
            gt_depths = sorted(Path(args.gt_depth).files("*.txt"))  # in *.txt
        else:
            print('gt of the dataset is not support')

        """ algin pred depth with gt depth """
        if args.dataset in ['tum']:
            # rgb_list = read_file_list(args.gt_depth + '/../rgb.txt')
            # depth_list = read_file_list(args.gt_depth + '/../depth.txt')
            # rgb_asso, depth_asso = associate(rgb_list, depth_list, 0, 0.02)
            # rgb_rm = [x for x, y in list(enumerate(rgb_list)) if y not in rgb_asso]
            # depth_rm = [x for x, y in list(enumerate(depth_list)) if y not in depth_asso]
            # # 按照rgb/depth rm index删除pred_depth/gt_depth中元素
            # pred_depths_algined = [pred_depths[i] for i in range(len(pred_depths)) if i not in rgb_rm] #仅仅保留不再rm中的索引元素
            # gt_depths = [gt_depths[i] for i in range(len(gt_depths)) if i not in depth_rm]

            # gt_list = [f.split('.')[0] for f in os.listdir(args.gt_depth) if f.endswith('png')]
            # pred_list = sorted([f.split('.')[0].zfill(10) for f in os.listdir(args.pred_depth)])
            # pred_rm = [x for x, y in list(enumerate(pred_list)) if y not in gt_list]
            # pred_depths_algined = [pred_depths[i] for i in range(len(pred_depths)) if i not in pred_rm]

            # gt_rm = [x for x, y in list(enumerate(gt_list)) if y not in pred_list]
            # gt_depths_algined = [gt_depths[i] for i in range(len(gt_depths)) if i not in gt_rm]

            rgb_list = read_file_list(args.gt_depth + '/../rgb.txt')
            depth_list = read_file_list(args.gt_depth + '/../depth.txt')
            rgb_asso, depth_asso = associate(rgb_list, depth_list, 0, 0.02)
            
            depth_rm = [x for x, y in list(enumerate(depth_list)) if y not in depth_asso]
            gt_depths = [gt_depths[i] for i in range(len(gt_depths)) if i not in depth_rm]

            # 按照rgb/depth rm index删除pred_depth/gt_depth中元素
            pred_list = [float(f.rsplit('.',1)[0]) for f in os.listdir(args.pred_depth)]

            rgb_asso_rm = [x for x, y in list(enumerate(rgb_asso)) if y not in pred_list]
            
            gt_depths_algined = [gt_depths[i] for i in range(len(gt_depths)) if i not in rgb_asso_rm]

            pred_depths_algined = pred_depths

            assert len(pred_depths_algined) == len(gt_depths_algined)


        elif args.dataset in ['euroc']:
            gt_list = [f.split('.')[0] for f in os.listdir(args.gt_depth) if f.endswith('txt')]
            pred_list = [f.split('.')[0] for f in os.listdir(args.pred_depth)]
            pred_depths_algined, overlapping = [], []
            for i in range(len(pred_list)):
                if pred_list[i] in gt_list:
                    pred_depths_algined.append(pred_depths[i])
                    overlapping.append(pred_list[i])

        elif args.dataset in ['kitti']:
            gt_list = [f.split('.')[0] for f in os.listdir(args.gt_depth) if f.endswith('png')]
            pred_list = sorted([f.split('.')[0].zfill(10) for f in os.listdir(args.pred_depth)])
            pred_rm = [x for x, y in list(enumerate(pred_list)) if y not in gt_list]
            pred_depths_algined = [pred_depths[i] for i in range(len(pred_depths)) if i not in pred_rm]

            gt_rm = [x for x, y in list(enumerate(gt_list)) if y not in pred_list]
            gt_depths_algined = [gt_depths[i] for i in range(len(gt_depths)) if i not in gt_rm]

        pred_depths = pred_depths_algined
        gt_depths = gt_depths_algined

        if len(gt_depths) != len(pred_depths_algined):
            assert(False and "Not evaluate all groundtruth data!")
            # min_len = min(len(gt_depths), len(pred_depths_algined))
            # gt_depths = gt_depths[:min_len]
            # pred_depths = pred_depths[:min_len]

        assert (len(pred_depths) == len(gt_depths))

        """ Get segmentation masks """
        seg_masks = None
        if args.seg_mask is not None:
            self.dynamic_colors = np.loadtxt(
                Path(args.seg_mask)/'dynamic_colors.txt').astype('uint8')
            seg_masks = sorted(Path(args.seg_mask).files("*.png"))

        self.evaluate_depth(gt_depths, pred_depths, seg_masks, eval_mono=True)

    def evaluate_depth(self, gt_depths, pred_depths, seg_masks=None, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths: list of gt depth files
            pred_depths: list of predicted depth files
            eval_mono (bool): use median scaling if True
        """
        full_errors = []
        static_errors = []
        dynamic_errors = []
        ratios = []

        print("==> Evaluating depth result: ", self.pred_depth)
        for i in tqdm(range(len(pred_depths))):
            # load gt depth and pred depth
            pred_depths[i] = np.load(pred_depths[i])
            
            if args.dataset in ['tum']:
                gt_depths[i] = imread(gt_depths[i]).astype(np.float32) / 5000
            elif args.dataset == 'kitti':
                gt_depths[i] = cv2.imread(gt_depths[i])[:,:,0]
            elif args.dataset == 'euroc':
                gt_depths[i] = np.loadtxt(gt_depths[i])
                
            else:
                print('the datset is not support')

            # load seg mask
            if seg_masks is not None:
                dynamic_mask = np.zeros_like(gt_depths[i])
                seg_mask = imread(seg_masks[i])
                for item in self.dynamic_colors:
                    cal_mask_0 = seg_mask[:, :, 0] == item[0]
                    cal_mask_1 = seg_mask[:, :, 1] == item[1]
                    cal_mask_2 = seg_mask[:, :, 2] == item[2]
                    cal_mask = cal_mask_0 * cal_mask_1 * cal_mask_2
                    dynamic_mask[cal_mask] = 1

            # gt
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            mask = np.logical_and(gt_depth > self.min_depth, 
                                  gt_depth < self.max_depth)

            # # resize predicted depth to gt resolution
            pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

            # pre-process
            if args.dataset == 'kitti':
                crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            elif args.dataset == 'nyu':
                crop = np.array([45, 471, 41, 601]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            val_pred_depth = pred_depth[mask]
            val_gt_depth = gt_depth[mask]

            # median scaling is used for monocular evaluation
            ratio = 1
            if eval_mono:
                ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                ratios.append(ratio)
                val_pred_depth *= ratio
                
            val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
            val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

            full_errors.append(compute_depth_errors(
                val_gt_depth, val_pred_depth))

            if seg_masks is not None:
                val_dynamic_mask = dynamic_mask[mask]

                # every image has static regions
                static_errors.append(compute_depth_errors(val_gt_depth[val_dynamic_mask == 0],
                                                          val_pred_depth[val_dynamic_mask == 0]))

                # note that some images may not have dynamic regions,
                # we only average results on images that have dynamic regions
                if (val_gt_depth[val_dynamic_mask == 1]).shape[0] > 0:
                    full_errors.append(compute_depth_errors(
                        val_gt_depth, val_pred_depth))
                    dynamic_errors.append(compute_depth_errors(val_gt_depth[val_dynamic_mask == 1],
                                                               val_pred_depth[val_dynamic_mask == 1]))

            pred_depths[i] = None

        if eval_mono:
            ratios = np.array(ratios)
            print(
                " Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))

        mean_errors_full = np.array(full_errors).mean(0)
        print("Evaluation on full images")
        print("\n " + ("{:>8} | " * 9).format("abs_rel", "sq_rel",
              "log10", "rmse", "rmse_log", "a1", "a2", "a3", "mae"))
        print(("&{: 8.3f}  " * 9).format(*mean_errors_full.tolist()) + "\\\\")

        # plot each evaluation results in the eight metrics
        # import matplotlib.pyplot as plt
        # x_values = list(range(1, len(full_errors)+1))
        # abs_rel = [t[0] for t in full_errors]
        # sq_rel = [t[1] for t in full_errors]
        # log10 = [t[2] for t in full_errors]
        # rmse = [t[3] for t in full_errors]
        # rmse_log = [t[4] for t in full_errors]
        # a1 = [t[5] for t in full_errors]
        # a2 = [t[6] for t in full_errors]
        # a3 = [t[7] for t in full_errors]

        # plt.plot(x_values, abs_rel, label='Abs Rel')
        # plt.ylim([0,0.7])
        # plt.legend()
        # plt.grid(True)
        # plt.show()


eval = DepthEval()
eval.main()
