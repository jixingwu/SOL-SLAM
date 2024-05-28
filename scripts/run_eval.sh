#!/bin/bash

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_finetune_r4/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r4/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r8/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r16/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r32/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r64/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02

# python tools/eval_depth_fixed.py   --dataset kitti --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/1/model_v1_refiner_r128/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/kitti_odom/odom_data/09/proj_depth/groundtruth/image_02


python tools/eval_depth_fixed.py --dataset tum --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/2/model_v1_refiner_r16/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/depth

python tools/eval_depth_fixed.py --dataset tum --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/2/model_v1_refiner_r32/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/depth

python tools/eval_depth_fixed.py --dataset tum --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/2/model_v1_refiner_r64/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/depth

python tools/eval_depth_fixed.py --dataset tum --pred_depth /media/jixingwu/medisk1/DF-VO/result/tmp/2/model_v1_refiner_r128/depth --gt_depth /media/jixingwu/medisk1/DF-VO/dataset/tum/rgbd_slam/rgbd_dataset_freiburg2_desk/depth



