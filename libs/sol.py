''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: DF-VO core program
'''

import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm

from libs.geometry.camera_modules import SE3
import libs.datasets as Dataset
from libs.deep_models.deep_models import DeepModel
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timer
from libs.matching.keypoint_sampler import KeypointSampler
from libs.matching.depth_consistency import DepthConsistency
from libs.tracker import EssTracker, PnpTracker
from libs.general.utils import *
from libs.geometry.ops_3d import *
from libs.geometry.triangulate import *
from matplotlib import pyplot as plt
from libs.deep_models.depth.datasets.train_folders import TrainBatch

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import PoseStamped
from queue import Queue
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType
from libs.deep_models.depth.visualization import *

class DFVO():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # configuration
        self.cfg = cfg

        # tracking stage
        self.tracking_stage = 0

        # predicted global poses
        self.global_poses = {0: SE3()}

        # validation data for online learning
        self.val_dict = None

        # reference data and current data
        self.initialize_data()

        self.setup()

        self.inputs_bs = []

        if self.cfg.save_pose or self.cfg.load_pose:
            self.pose_path = os.path.join(self.cfg.directory.pose_dir, self.cfg.dataset, self.cfg.seq)
            self.sparse_depth_path = os.path.join(self.cfg.directory.sparse_depth_dir, self.cfg.dataset, self.cfg.seq)


        # ros setup
        self.pub = {} # rgb, depth, depth color etc.
        self.odom_queue, self.depth_queue, self.time_queue = Queue(), Queue(), Queue()
        self.cv_bridge = CvBridge()
        self.ros_setup()

    def setup(self):
        """Reading configuration and setup, including

            - Timer
            - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        # timer
        self.timers = Timer(self.cfg)

        # intialize dataset
        self.dataset = Dataset.datasets[self.cfg.dataset](self.cfg)
        
        # get tracking method
        self.tracking_method = self.cfg.tracking_method
        self.initialize_tracker()

        # initialize keypoint sampler
        self.kp_sampler = KeypointSampler(self.cfg)
        
        # Deep networks
        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()
        if self.cfg.online_finetune.enable:
            self.deep_models.setup_train()
        
        # Depth consistency
        if self.cfg.kp_selection.depth_consistency.enable:
            self.depth_consistency_computer = DepthConsistency(self.cfg, self.dataset.cam_intrinsics)

        # visualization interface
        self.drawer = FrameDrawer(self.cfg.visualization)

        # validate load
        if self.cfg.online_learning.enable and self.cfg.online_learning.val_enable:
            self.val_dict = self.dataset.load_val_data(self.deep_models.valid_transform)

    def ros_setup(self):
        rospy.init_node('so_slam')
        self.pub['image'] = rospy.Publisher('/so_slam/image/image_raw', ImageMsg, queue_size=10)
        self.pub['depth'] = rospy.Publisher('/so_slam/depth/image_raw', ImageMsg, queue_size=10)
        self.pub['depth_color'] = rospy.Publisher('/so_slam/depth_color/image_raw', ImageMsg, queue_size=10)

        if self.cfg.save_pose and not self.cfg.load_pose:
            rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.odom_callback)
            rospy.Subscriber('/orb_slam3/sparse_depth_image', ImageMsg, self.depth_callback)
        elif self.cfg.load_pose and not self.cfg.save_pose:
            print("==> Load pose and sparse depth from npy files")
        else:
            assert False and "Error: save_pose and load_pose are both True or False!!"



    def publish(self, name, t, id, data):
        if name in ['image', 'depth_color', 'dense_depth_color']:
            msg = self.cv_bridge.cv2_to_imgmsg(data.astype(np.uint8), encoding="rgb8")
        if name in ['depth']:
            data = np.array(data, dtype=np.float32)
            msg = self.cv_bridge.cv2_to_imgmsg(data, encoding="passthrough")

        msg.header.frame_id = name
        msg.header.seq = id
        msg.header.stamp = rospy.Time.from_sec(t)
        self.pub[name].publish(msg)

    def odom_callback(self, msg):
        assert not self.cfg.load_pose and "Error: load_pose is True!!"

        t, pose = msg.header.stamp.to_sec(), msg.pose
        loc = np.array([pose.position.x, pose.position.y, pose.position.z])
        rot = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.from_quat(rot).as_matrix(), loc
        self.odom_queue.put((t, T))
        self.time_queue.put(t)

        # print("Received pose at time: {}, current time: {}".format(t, self.cur_data['timestamp']))
        if abs(t - self.cur_data['timestamp']) < 0.01:
            self.cur_data['pose'] = SE3(T)
            if self.cfg.save_pose:
                if not os.path.exists(self.pose_path):
                    os.makedirs(self.pose_path)
                np.save(os.path.join(self.pose_path, "{:010d}.npy".format(self.cur_data['id'])), T)
        else:
            assert False and "Pose and image timestamp mismatch!"

    def depth_callback(self, msg):
        assert not self.cfg.load_pose and "Error: load_pose is True!!"

        t = msg.header.stamp.to_sec()
        depth = np.array(self.cv_bridge.imgmsg_to_cv2(msg, "passthrough"))
        self.depth_queue.put((t, np.abs(depth)))

        # print("Received depth at time: {}, current time: {}".format(t, self.cur_data['timestamp']))
        if abs(t - self.cur_data['timestamp']) < 0.01:
            self.cur_data['sparse_depth'] = np.abs(depth)
            if self.cfg.save_pose:
                if not os.path.exists(self.sparse_depth_path):
                    os.makedirs(self.sparse_depth_path)
                np.save(os.path.join(self.sparse_depth_path, "{:010d}.npy".format(self.cur_data['id'])), depth)
        else:
            assert False and "Depth and image timestamp mismatch!"

    def initialize_data(self):
        """initialize data of current view and reference view
        """
        self.ref_data = {}
        self.cur_data = {}
        self.rel_pose = SE3()
        self.keyframe_list = []
        self.bs = 0

    def initialize_tracker(self):
        """Initialize tracker
        """
        if self.tracking_method == 'hybrid':
            self.e_tracker = EssTracker(self.cfg, self.dataset.cam_intrinsics, self.timers)
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'deep_pose':
            return
        else:
            assert False, "Wrong tracker is selected, choose from [hybrid, PnP, deep_pose]"

    def update_global_pose(self, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])

    def tracking(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation and translation direction;
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails
        """
        # First frame
        if self.tracking_stage == 0:
            # initial pose
            if self.cfg.directory.gt_pose_dir is not None:
                self.cur_data['pose'] = SE3(self.dataset.gt_poses[self.cur_data['id']])
            else:
                self.cur_data['pose'] = SE3()
            self.ref_data['motion'] = SE3()
            return

        # Second to last frames
        elif self.tracking_stage >= 1:
            ''' keypoint selection '''
            if self.tracking_method in ['hybrid', 'PnP']:
                # Depth consistency (CNN depths + CNN pose)
                if self.cfg.kp_selection.depth_consistency.enable:
                    self.depth_consistency_computer.compute(self.cur_data, self.ref_data)

                # kp_selection
                self.timers.start('kp_sel', 'tracking')
                kp_sel_outputs = self.kp_sampler.kp_selection(self.cur_data, self.ref_data)
                if kp_sel_outputs['good_kp_found']:
                    self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)
                self.timers.end('kp_sel')

            ''' Pose estimation '''
            # Initialize hybrid pose
            hybrid_pose = SE3()
            E_pose = SE3()

            if not(kp_sel_outputs['good_kp_found']):
                print("No enough good keypoints, constant motion will be used!")
                pose = self.ref_data['motion']
                self.update_global_pose(pose, 1)
                return 


            ''' E-tracker '''
            if self.tracking_method in ['hybrid']:
                # Essential matrix pose
                self.timers.start('E-tracker', 'tracking')
                e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.kp_src],
                                self.cur_data[self.cfg.e_tracker.kp_src],
                                not(self.cfg.e_tracker.iterative_kp.enable)) # pose: from cur->ref
                E_pose = e_tracker_outputs['pose']
                self.timers.end('E-tracker')

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers
                self.ref_data['inliers'] = e_tracker_outputs['inliers']

                # scale recovery
                if np.linalg.norm(E_pose.t) != 0:
                    self.timers.start('scale_recovery', 'tracking')
                    scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, False)
                    scale = scale_out['scale']
                    if self.cfg.scale_recovery.kp_src == 'kp_depth':
                        self.cur_data['kp_depth'] = scale_out['cur_kp_depth']
                        self.ref_data['kp_depth'] = scale_out['ref_kp_depth']
                        self.cur_data['rigid_flow_mask'] = scale_out['rigid_flow_mask']
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('scale_recovery')

                # Iterative keypoint refinement
                if np.linalg.norm(E_pose.t) != 0 and self.cfg.e_tracker.iterative_kp.enable:
                    self.timers.start('E-tracker iter.', 'tracking')
                    # Compute refined keypoint
                    self.e_tracker.compute_rigid_flow_kp(self.cur_data,
                                                         self.ref_data,
                                                         hybrid_pose)

                    e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                self.cur_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                True) # pose: from cur->ref
                    E_pose = e_tracker_outputs['pose']

                    # Rotation
                    hybrid_pose.R = E_pose.R

                    # save inliers
                    self.ref_data['inliers'] = e_tracker_outputs['inliers']

                    # scale recovery
                    if np.linalg.norm(E_pose.t) != 0 and self.cfg.scale_recovery.iterative_kp.enable:
                        scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, True)
                        scale = scale_out['scale']
                        if scale != -1:
                            hybrid_pose.t = E_pose.t * scale
                    else:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('E-tracker iter.')

            ''' PnP-tracker '''
            if self.tracking_method in ['PnP', 'hybrid']:
                # PnP if Essential matrix fail
                if np.linalg.norm(E_pose.t) == 0 or scale == -1:
                    self.timers.start('pnp', 'tracking')
                    pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.kp_src],
                                    self.ref_data['depth'],
                                    not(self.cfg.pnp_tracker.iterative_kp.enable)
                                    ) # pose: from cur->ref
                    
                    # Iterative keypoint refinement
                    if self.cfg.pnp_tracker.iterative_kp.enable:
                        self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
                        pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.ref_data['depth'],
                                    True
                                    ) # pose: from cur->ref

                    self.timers.end('pnp')

                    # use PnP pose instead of E-pose
                    hybrid_pose = pnp_outputs['pose']
                    self.tracking_mode = "PnP"

            ''' Summarize data '''
            # update global poses
            self.rel_pose = hybrid_pose
            self.ref_data['pose'] = copy.deepcopy(hybrid_pose)
            self.ref_data['motion'] = copy.deepcopy(hybrid_pose)
            pose = self.ref_data['pose']
            self.update_global_pose(pose, 1)

    def update_data(self, ref_data, cur_data):
        """Update data
        
        Args:
            ref_data (dict): reference data
            cur_data (dict): current data
        
        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'] = cur_data['id']
            else:
                if ref_data.get(key, -1) is -1:
                    ref_data[key] = {}
                ref_data[key] = cur_data[key]
        
        # Delete unused flow to avoid data leakage
        ref_data['flow'] = None
        cur_data['flow'] = None
        ref_data['flow_diff'] = None
        return ref_data, cur_data

    def load_raw_data(self):
        """load image data and (optional) GT/precomputed depth data
        """

    
    def deep_model_inference(self):
        """deep model prediction
        """
        if self.tracking_method in ['hybrid', 'PnP']:
            # Single-view Depth prediction

            img = self.cur_data['img']
            id = self.cur_data['id']
            timestamp = self.cur_data['timestamp']

            self.cur_data['raw_depth'] = self.deep_models.forward_depth(img, timestamp)
            self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                (self.cfg.image.width, self.cfg.image.height),
                                                interpolation=cv2.INTER_NEAREST)

            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # Two-view flow
            if self.tracking_stage >= 1:
                self.timers.start('flow_cnn', 'deep inference')
                if self.cfg.off_flownet:
                    flow_id = self.cur_data['id']
                    flow_file = os.path.join(self.cfg.directory.flow_dir, self.cfg.dataset, self.cfg.seq, "{:010d}.npy".format(flow_id))
                    flows = np.load(flow_file, allow_pickle=True).item()

                else:
                    flows = self.deep_models.forward_flow(
                                            self.cur_data,
                                            self.ref_data,
                                            forward_backward=self.cfg.deep_flow.forward_backward)

                # Store flow
                self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
                if self.cfg.deep_flow.forward_backward:
                    self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                    self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()
                
                self.timers.end('flow_cnn')

    def flownet_inference_to_npy(self):
        """flow net inference and save results to npy files
        """
        print("==> FlowNet inference and save results to npy files")

        flow_path = os.path.join(self.cfg.directory.flow_dir, self.cfg.dataset, self.cfg.seq)
        if not os.path.exists(flow_path):
            os.makedirs(flow_path)
        print("==> Save flow results to {}".format(flow_path))

        for img_id in tqdm(range(0, len(self.dataset))):
            # Read image data
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)
            self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

            # Deep model inferences
            if self.tracking_stage >= 1:
                flows = self.deep_models.forward_flow(self.cur_data,
                                                      self.ref_data,
                                                      forward_backward=self.cfg.deep_flow.forward_backward)
                # Save all flows
                np.save(flow_path + "/{:010d}.npy".format( self.cur_data['id']), flows)

            self.tracking_stage += 1
            self.update_data(self.ref_data, self.cur_data)

    def callback_data_to_npy(self):
        """ Save pose and sparse to npy files
        """
        if self.cfg.save_pose:
            print("==> Start run save pose and sparse depth")
            print("==> Running sequence: {}".format(self.cfg.seq))
        else:
            assert False and "Error: save_pose is False!!"

        if self.cfg.no_confirm:
            start_frame = 0
        else:
            start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len(self.dataset), self.cfg.frame_step)):
            # Initialize ids and timestamps
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

            # Read image data
            self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

            self.deep_model_inference()

            self.publish('image', self.cur_data['timestamp'], self.cur_data['id'], self.cur_data['img'])
            self.publish('depth', self.cur_data['timestamp'], self.cur_data['id'], self.cur_data['raw_depth'])

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_data(
                                    self.ref_data,
                                    self.cur_data,
            )

            self.tracking_stage += 1

        print("=> Finish!")
        print("The results are saved in [{}], [{}].".format(self.pose_path, self.sparse_depth_path))

    def densify_sparse_depth(self, grid_num=20, visualize=False):
        """ Densify sparse depth
        """
        sparse_depth = self.cur_data['sparse_depth']
        seg_masks = self.cur_data['seg']
        # 图像尺寸
        height, width = sparse_depth.shape

        # 初始化稠密深度图
        dense_depth = np.zeros((height, width))

        # 计算每行，列的格子数
        grid_height = height // grid_num
        grid_width = width // grid_num

        ''' 遍历每个格, 如果格子全属于同一个mask id, 则填充该格子的深度均值
            如果格子不是全属于一个mask id，则根据mask id所包含的深度均值进行填充 '''
        for i in range(grid_num):
            for j in range(grid_num):
                # 获取当前网格的深度值和mask id
                grid_depth = sparse_depth[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
                grid_mask = seg_masks[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]

                # 若网格内没有深度值，则跳过
                if np.sum(grid_depth) == 0:
                    continue

                # 获取当前网格内的所有不同的mask id
                unique_mask_ids = np.unique(grid_mask)

                if len(unique_mask_ids) == 1:
                    # 如果网格全属于同一个mask id，填充该网格的深度均值
                    dense_depth[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width] = np.mean(grid_depth[grid_depth > 0])
                else:
                    # 如果网格不全属于同一个mask id，根据mask id所包含的深度均值进行填充
                    for mask_id in unique_mask_ids:
                        mask_depth = grid_depth[grid_mask == mask_id]
                        if np.sum(mask_depth) == 0:
                            continue
                        mask_mean_depth = np.mean(mask_depth[mask_depth > 0])
                        dense_depth[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width][grid_mask == mask_id] = mask_mean_depth
        # 可视化dense depth
        if visualize:
            # plot grid mesh, sparse depth and dense depth
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(self.cur_data['seg_color'])
            for i in range(1, grid_num):
                plt.plot([i*grid_width, i*grid_width], [0, height], 'r', linewidth=0.5)
            for j in range(1, grid_num):
                plt.plot([0, width], [j*grid_height, j*grid_height], 'r', linewidth=0.5)
            
            # 获取sparse_depth中非零元素的坐标，使用scatter函数将这些点画在图上   
            y, x = np.nonzero(sparse_depth)
            plt.scatter(x, y, color='white', s=10)
            plt.title("Grid Mesh")

            plt.subplot(132)
            dense_depth_tmp1 = dense_depth.copy()
            plt.imshow(dense_depth_tmp1, cmap='viridis')
            plt.title("Dense Depth1")
        
        ''' 二次填充，对于没有深度值的像素，用其周围像素的深度均值填充 '''
        from scipy.spatial import KDTree
        # # 获取所有深度值不为0的像素的坐标和深度值
        # non_zero_depth_indices = np.transpose(np.nonzero(dense_depth))
        # non_zero_depth_values = dense_depth[dense_depth > 0]

        # # 创建KDTree
        # tree = KDTree(non_zero_depth_indices)

        # # 获取所有深度值为0的像素的坐标
        # zero_depth_indices = np.transpose(np.where(dense_depth == 0))

        # # 找到每个没有深度值的像素最近的n个有深度值的像素
        # n = 5  # 可以根据需要调整
        # distances, indices = tree.query(zero_depth_indices, k=n)

        # # 计算这些像素的深度值的平均值
        # mean_values = np.mean(non_zero_depth_values[indices], axis=1)

        # # 将计算得到的深度值赋值给深度值为0的像素
        # dense_depth[dense_depth == 0] = mean_values

        # 获取所有不同的mask id
        unique_mask_ids = np.unique(seg_masks)
        for mask_id in unique_mask_ids:
            # 获取当前mask id下深度值不为0的像素的坐标和深度值
            non_zero_depth_indices = np.transpose(np.nonzero((dense_depth > 0) & (seg_masks == mask_id)))
            non_zero_depth_values = dense_depth[(dense_depth > 0) & (seg_masks == mask_id)]

            # 如果non_zero_depth_indices为空，则跳过当前循环
            if len(non_zero_depth_indices) <= 0:
                continue

            # 创建KDTree
            tree = KDTree(non_zero_depth_indices)

            # 获取当前mask id下深度值为0的像素的坐标
            zero_depth_indices = np.transpose(np.where((dense_depth == 0) & (seg_masks == mask_id)))

            n = 5  # 可以根据需要调整
            if len(zero_depth_indices) > 0 and len(non_zero_depth_indices) > n:
                # 找到每个没有深度值的像素最近的n个有深度值的像素
                distances, indices = tree.query(zero_depth_indices, k=min(n, len(non_zero_depth_indices)))

                # 计算这些像素的深度值的平均值
                mean_values = np.mean(non_zero_depth_values[indices], axis=1)
            else:
                mean_values = non_zero_depth_values.mean()
            
            # 将计算得到的深度值赋值给深度值为0的像素
            dense_depth[(dense_depth == 0) & (seg_masks == mask_id)] = mean_values


        if visualize:
            plt.subplot(133)
            # 将dense_depth中为0的值替换为nan
            dense_depth_tmp2 = np.where(dense_depth==0, np.nan, dense_depth)
            plt.imshow(dense_depth_tmp2, cmap='viridis')
            plt.title("Dense Depth2")
            plt.show()
        
        # save several data
        # np.save(f"/media/jixingwu/medisk1/DF-VO/tools/draw_densify_sparse_depth/{self.cur_data['id']}_sparse_depth.npy", sparse_depth)
        # np.save(f"/media/jixingwu/medisk1/DF-VO/tools/draw_densify_sparse_depth/{self.cur_data['id']}_dense_depth_tmp1.npy", dense_depth_tmp1)
        # np.save(f"/media/jixingwu/medisk1/DF-VO/tools/draw_densify_sparse_depth/{self.cur_data['id']}_dense_depth_tmp2.npy", dense_depth_tmp2)
        dense_depth_color = plt.get_cmap

        return dense_depth

    def prepare_training_data(self, transform):
        """Prepare training data for online learning
        """
        rel_poses = [frame['pose'] for frame in self.keyframe_list]

        ref_imgs = [frame['img'] for frame in self.keyframe_list]
        ref_segs = [frame['seg'] for frame in self.keyframe_list]

        cur_img = self.cur_data['img']
        cur_seg = self.cur_data['seg']
        cur_seg_id = self.cur_data['seg_id']

        pseudo_dense_depth = self.densify_sparse_depth(visualize=False)

        self.cur_data['pseu_depth'] = pseudo_dense_depth

        inputs = (ref_imgs, cur_img, rel_poses, pseudo_dense_depth,
                  cur_seg, ref_segs, self.dataset.cam_intrinsics.mat, cur_seg_id)

        inputs = TrainBatch(inputs, transform).getitem()
        self.bs += 1
        return inputs

    def append_keyframe_list(self):
        frame = {'id': self.cur_data['id'],
                 'img': self.cur_data['img'],
                 'seg': self.cur_data['seg'],
                 'seg_id': self.cur_data['seg_id'],
                 'pose': self.rel_pose.pose}

        self.keyframe_list.append(frame)

    def main(self):
        """Main program
        """
        print("==> Start DF-VO")
        print("==> Running sequence: {}".format(self.cfg.seq))

        if self.cfg.no_confirm:
            start_frame = 0
        else:
            start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len(self.dataset), self.cfg.frame_step)):
            """ Data reading """
            # Initialize ids and timestamps
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

            # load pose and sparse depth
            try:
                if self.cfg.load_pose and not self.cfg.save_pose:
                    if self.cfg.dataset == 'kitti':
                        pose = np.load(os.path.join(self.pose_path, "{:010d}.npy".format(self.cur_data['id'])))
                        sparse_depth = np.load(os.path.join(self.sparse_depth_path, "{:010d}.npy".format(self.cur_data['id'])))
                    elif 'tum' in self.cfg.dataset:
                        pose = np.load(os.path.join(self.pose_path, "{:.6f}.npy".format(self.cur_data['timestamp'])))
                        sparse_depth = np.load(os.path.join(self.sparse_depth_path, "{:.6f}.npy".format(self.cur_data['timestamp'])))
                    self.cur_data['pose'] = SE3(pose)
                    self.cur_data['sparse_depth'] = sparse_depth
            except Exception:
                print("Error: pose or sparse depth not found!")
                """ Update reference and current data """
                self.ref_data, self.cur_data = self.update_data(
                                        self.ref_data,
                                        self.cur_data,
                )

                self.tracking_stage += 1
                continue
            
            self.timers.start('DF-VO')
            self.tracking_mode = "Ess. Mat."

            # Read image data
            self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

            # Read segment data
            self.cur_data['seg'], self.cur_data['seg_color'], self.cur_data['seg_id'] = self.dataset.get_segment(self.cur_data['id'])

            # Deep model inferences
            self.timers.start('deep_inference')
            self.deep_model_inference()
            self.timers.end('deep_inference')

            self.publish('image', self.cur_data['timestamp'], self.cur_data['id'], self.cur_data['img'])
            self.publish('depth', self.cur_data['timestamp'], self.cur_data['id'], self.cur_data['raw_depth'])
            vis = visualize_depth(self.cur_data['raw_depth'], cv2.COLORMAP_MAGMA).permute(1, 2, 0).numpy() * 255
            self.publish('depth_color', self.cur_data['timestamp'], self.cur_data['id'], vis)

            """ Visual odometry """
            self.timers.start('tracking')
            self.tracking()
            self.timers.end('tracking')

            """ Online Learning """
            if self.tracking_stage >= 1 and self.cfg.online_learning.enable and np.any(self.cur_data['sparse_depth']):

                if len(self.keyframe_list) == self.cfg.online_learning.max_size:
                    inputs = self.prepare_training_data(transform=self.deep_models.train_transform)
                    self.inputs_bs.append(inputs)

                # self.keyframe_list.append((self.cur_data, self.rel_pose.pose))
                self.append_keyframe_list()

                if len(self.keyframe_list) > self.cfg.online_learning.max_size:
                    self.keyframe_list = self.keyframe_list[-self.cfg.online_learning.max_size:]

                if self.bs == self.cfg.online_learning.batch_size:
                    self.timers.start('online learning')
                    self.deep_models.depth.run_epoch(self.inputs_bs)
                    self.timers.end('online learning')
                    self.bs = 0
                    self.inputs_bs = []

            """ Validation """
            if self.cfg.online_learning.enable and self.cfg.online_learning.val_enable and \
                    self.deep_models.depth.step % self.cfg.online_learning.val_step == 0 and \
                    self.val_dict is not None:
                self.deep_models.depth.val(self.val_dict)

            """ Visualization """
            if self.cfg.visualization.enable:
                self.timers.start('visualization')
                self.drawer.main(self)
                self.timers.end('visualization')

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_data(
                                    self.ref_data,
                                    self.cur_data,
            )

            self.tracking_stage += 1

            self.timers.end('DF-VO')

        print("=> Finish!")

        """ Display & Save result """
        print("The result is saved in [{}].".format(self.cfg.directory.result_dir))
        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map.png".format(self.cfg.directory.result_dir)
        cv2.imwrite(map_png, self.drawer.data['traj'])

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.directory.result_dir, self.cfg.seq)
        self.dataset.save_result_traj(traj_txt, self.global_poses)

        # save finetuned model
        # if self.cfg.online_finetune.enable and self.cfg.online_finetune.save_model:
        #     self.deep_models.save_model()

        # Output experiement information
        self.timers.time_analysis()
        self.timers.time_analysis_to_txt()
