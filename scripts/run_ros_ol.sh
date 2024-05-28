#!/bin/bash

echo "fintune"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r2"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r2.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r4"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r4.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r8"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r8.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r16"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r16.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r32"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r32.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID

echo "refiner r64"
# 启动 roslaunch 进程并获取其 PID
roslaunch orb_slam3_ros tum_rgbd_monodepth.launch >> result/tmp/0/roslaunch_output.txt &
ROSLAUNCH_PID=$!
# 启动 python 进程并等待其结束
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/ol_r64.yml --seq rgbd_dataset_freiburg2_desk --no_confirm --off_flownet >> result/tmp/0/python_output.txt
# 当 python 进程结束后，杀死 roslaunch 进程
kill $ROSLAUNCH_PID