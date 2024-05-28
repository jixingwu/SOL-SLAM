import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import PoseStamped

import numpy as np
import os
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


cv_bridge = CvBridge()
pose_path = "/media/jixingwu/medisk1/DF-VO/dataset/tum/pose_data/tum-2/rgbd_dataset_freiburg2_desk"
depth_path = "/media/jixingwu/medisk1/DF-VO/dataset/tum/sparse_depth/tum-2/rgbd_dataset_freiburg2_desk"

def odom_callback(msg):
    t, pose = msg.header.stamp.to_sec(), msg.pose
    loc = np.array([pose.position.x, pose.position.y, pose.position.z])
    rot = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.from_quat(rot).as_matrix(), loc
    
    np.save(os.path.join(pose_path, "{:.6f}.npy".format(t)), T)


def depth_callback(msg):
    t = msg.header.stamp.to_sec()
    depth = np.array(cv_bridge.imgmsg_to_cv2(msg, "passthrough"))
    
    np.save(os.path.join(depth_path, "{:.6f}.npy".format(t)), depth)


if __name__ == '__main__':
    rospy.init_node('so_slam_tools')
    rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, odom_callback)
    rospy.Subscriber('/orb_slam3/sparse_depth_image', ImageMsg, depth_callback)
    rospy.spin()