#!/bin/bash

# Get the first command line argument
input=$1

if [ "$input" = "kitti" ]
then
  echo "Running KITTI datasets"
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/off.yml --no_confirm --seq 10
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/off.yml --no_confirm --seq 08
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/off.yml --no_confirm --seq 07
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/off.yml --no_confirm --seq 05
elif [ "$input" = "tum" ]
then
  echo "Running TUM RGB-D datasets"
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/tum_config/off_flownet-tum2.yml --no_confirm --seq rgbd_dataset_freiburg2_desk
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/tum_config/off_flownet-tum2.yml --no_confirm --seq rgbd_dataset_freiburg2_pioneer_360
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/tum_config/off_flownet-tum2.yml --no_confirm --seq rgbd_dataset_freiburg2_pioneer_slam
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/tum_config/off_flownet-tum3.yml --no_confirm --seq rgbd_dataset_freiburg3_long_office_household_validation
    python apis/off-flownet.py -d options/examples/default_configuration.yml -c options/examples/tum_config/off_flownet-tum3.yml --no_confirm --seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop
else
    echo "Invalid input. Please enter 'kitti'."
fi