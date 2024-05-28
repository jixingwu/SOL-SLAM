#!/bin/bash

# KITTI
# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r4.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r8.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r16.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r32.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r64.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/refiner/r128.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/kitti_config/ol.yml --seq 09 --no_confirm --off_flownet --load_pose >> /media/jixingwu/medisk1/DF-VO/result/tmp/1/results.txt


# TUM
# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r2.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r4.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r8.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r16.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r32.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r64.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r128.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

# python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r256.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r512.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose

python apis/run.py -d options/examples/default_configuration.yml -c  options/examples/tum_config/ol_r1024.yml  --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet --load_pose