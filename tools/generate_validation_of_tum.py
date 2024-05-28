import sys
import os
import numpy
import argparse

from evaluation.tum_tool.associate import associate, read_file_list

# Argument Parsing
parser = argparse.ArgumentParser(description='VO system')
parser.add_argument("--seq", default=None, help="sequence")
parser.add_argument("--path", default=None, help="root path")
args = parser.parse_args()

rgb_d_pair = {}
data_dir = {}

# get image data directory
img_seq_dir = os.path.join(args.path, args.seq)
data_dir['img'] = os.path.join(img_seq_dir, "rgb")

# associate rgb-depth-pose timestamp pair
rgb_list = read_file_list(data_dir['img'] + '/../rgb.txt')
depth_list = read_file_list(data_dir['img'] + '/../depth.txt')
pose_list = read_file_list(data_dir['img'] + '/../groundtruth.txt')

for i in rgb_list:
    rgb_d_pair[i] = {}

# associate depth
matches = associate(rgb_list, depth_list, 0, 0.02)

for match in matches:
    rgb_stamp = match[0]
    depth_stamp = match[1]
    rgb_d_pair[rgb_stamp]['depth'] = depth_stamp

# val folder
data_dir['val'] = os.path.join(args.path, "val")
if not os.path.exists(data_dir['val']):
    os.makedirs(data_dir['val'])

if not os.path.exists(os.path.join(data_dir['val'], 'color')):
    os.makedirs(os.path.join(data_dir['val'], 'color'))

if not os.path.exists(os.path.join(data_dir['val'], 'depth')):
    os.makedirs(os.path.join(data_dir['val'], 'depth'))

# select 10% of the data as validation
val_color_list, val_depth_list = [], []
keys, values = list(rgb_list.keys()), list(rgb_list.values())

for i in range(0, len(rgb_list), 10):
    val_color_list.append(keys[i])
    val_depth_list.append(rgb_d_pair[keys[i]]['depth'])

# copy rgb and depth to color and depth folders
for i in val_color_list:
    os.system('cp ' + os.path.join(data_dir['img'], i) + ' ' + os.path.join(data_dir['val'], 'color'))
for i in val_depth_list:
    os.system('cp ' + os.path.join(data_dir['img'], rgb_d_pair[i]['depth']) + ' ' + os.path.join(data_dir['val'], 'depth'))