import configargparse
import numpy as np

class Paraser:
    def __init__(self):
        self.dataset_name = 'kitti'  # ['kitti', 'nyu', 'ddad', 'bonn', 'tum', 'rw', 'tum-2', 'tum-3']
        self.skip_frames = 1
        self.model_version = 'v1'  # ['v1', 'v2', 'v3']
        self.resnet_layers = 18
        self.log_path = 'logs'
        self.photo_weight = 1.0
        self.geometry_weight = 0.1
        self.smooth_weight = 0.1
        self.mask_rank_weight = 0.1
        self.normal_matching_weight = 0.1
        self.normal_rank_weight = 0.1
        self.exp_name = 'kitti_sc1'
        self.batch_size = 2
        self.epoch_size = 1
        self.num_epochs = 10
        self.lr = 1e-4
        self.scheduler_step_size = 1000

def get_training_size(dataset_name):

    if dataset_name == 'kitti_odom':
        training_size = [256, 832] #[370, 1226]
    elif dataset_name == 'ddad':
        training_size = [384, 640]
    elif dataset_name in ['nyu', 'tum', 'bonn', 'tum-2', 'tum-3']:
        training_size = [256, 320]
    elif dataset_name == 'rw': #[2048, 1536]
        training_size = [256, 192]
    else:
        print('unknown dataset type')

    return training_size
