#!/usr/bin/python3
import logging
import os

import numpy as np
import rospkg
import rospy
import tf
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config

class SimNode:
    def __init__(self, ns=""):
        print("[SimNode::__init__] START")

        rospy.init_node("mobiman_jackal_jaco")
        rospack = rospkg.RosPack()
        path = rospack.get_path("igibson-ros")
        config_filename = os.path.join(path, "config/mobiman_jackal_jaco.yaml")
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        self.config = parse_config(config_data)

        mode = self.config["mode"]
        action_timestep = self.config["action_timestep"]
        physics_timestep = self.config["physics_timestep"]
        render_timestep = self.config["render_timestep"]
        image_width = self.config["image_width"]
        image_height = self.config["image_height"]
        use_pb_gui = self.config["use_pb_gui"]

        print("[SimNode::__init__] config_data: " + str(config_data))
        print("[SimNode::__init__] mode: " + str(mode))
        print("[SimNode::__init__] action_timestep: " + str(action_timestep))
        print("[SimNode::__init__] physics_timestep: " + str(physics_timestep))
        print("[SimNode::__init__] render_timestep: " + str(render_timestep))
        print("[SimNode::__init__] image_width: " + str(image_width))
        print("[SimNode::__init__] image_height: " + str(image_height))
        print("[SimNode::__init__] use_pb_gui: " + str(use_pb_gui))

        self.ns = ns

        self.cmdx = 0.0
        self.cmdy = 0.0
        self.cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.image_pub = rospy.Publisher("gibson_ros/camera/rgb/image", ImageMsg, queue_size=10)
        self.depth_pub = rospy.Publisher("gibson_ros/camera/depth/image", ImageMsg, queue_size=10)
        self.lidar_pub = rospy.Publisher("gibson_ros/lidar/points", PointCloud2, queue_size=10)
        self.depth_raw_pub = rospy.Publisher("gibson_ros/camera/depth/image_raw", ImageMsg, queue_size=10)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        self.gt_pose_pub = rospy.Publisher("ground_truth_odom", Odometry, queue_size=10)
        self.camera_info_pub = rospy.Publisher("gibson_ros/camera/depth/camera_info", CameraInfo, queue_size=10)
        self.joint_states_pub = rospy.Publisher("gibson_ros/joint_states", JointState, queue_size=10)

        self.last_update = rospy.Time.now()
        rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.cmd_callback)
        rospy.Subscriber("/reset_pose", PoseStamped, self.tp_robot_callback)

        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        self.env = iGibsonEnv(
            config_file=config_data, 
            mode=mode, 
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            use_pb_gui=use_pb_gui,
            ros_node_init=True            
        )  # assume a 30Hz simulation
        self.env.reset()

        self.tp_time = None

        #print("[SimNode::__init__] DEBUG_INF")
        #while(1):
        #    continue

        ### NUA TODO: Add objects and try it again. 
        #object_states_keys = self.env.scene.object_states.keys()
        #print("[SimNode::__init__] object_states_keys: ")
        #print(object_states_keys)

        print("[SimNode::__init__] END")
        
    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            if (now - self.last_update).to_sec() > 2:
                cmdx = 0.0
                cmdy = 0.0
                cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                cmdx = self.cmdx
                cmdy = self.cmdy
                cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            obs, _, _, _ = self.env.step(cmd)

            '''
            rgb = (obs["rgb"] * 255).astype(np.uint8)
            normalized_depth = obs["depth"].astype(np.float32)
            depth = normalized_depth * self.env.sensors["vision"].depth_high
            depth_raw_image = (obs["depth"] * 1000).astype(np.uint16)

            image_message = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            depth_message = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
            depth_raw_message = self.bridge.cv2_to_imgmsg(depth_raw_image, encoding="passthrough")

            image_message.header.stamp = now
            depth_message.header.stamp = now
            depth_raw_message.header.stamp = now
            image_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_raw_message.header.frame_id = self.ns + "camera_depth_optical_frame"

            self.image_pub.publish(image_message)
            self.depth_pub.publish(depth_message)
            self.depth_raw_pub.publish(depth_raw_message)

            msg = CameraInfo(
                height=256,
                width=256,
                distortion_model="plumb_bob",
                D=[0.0, 0.0, 0.0, 0.0, 0.0],
                K=[128, 0.0, 128, 0.0, 128, 128, 0.0, 0.0, 1.0],
                R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                P=[128, 0.0, 128, 0.0, 0.0, 128, 128, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            msg.header.stamp = now
            msg.header.frame_id = self.ns + "camera_depth_optical_frame"
            self.camera_info_pub.publish(msg)

            if (self.tp_time is None) or ((self.tp_time is not None) and ((rospy.Time.now() - self.tp_time).to_sec() > 1.0)):
                scan = obs["scan"]
                lidar_header = Header()
                lidar_header.stamp = now
                lidar_header.frame_id = self.ns + "scan_link"

                laser_linear_range = self.env.sensors["scan_occ"].laser_linear_range
                laser_angular_range = self.env.sensors["scan_occ"].laser_angular_range
                min_laser_dist = self.env.sensors["scan_occ"].min_laser_dist
                n_horizontal_rays = self.env.sensors["scan_occ"].n_horizontal_rays

                laser_angular_half_range = laser_angular_range / 2.0
                angle = np.arange(
                    -np.radians(laser_angular_half_range),
                    np.radians(laser_angular_half_range),
                    np.radians(laser_angular_range) / n_horizontal_rays,
                )
                unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
                lidar_points = unit_vector_laser * (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

                lidar_message = pc2.create_cloud_xyz32(lidar_header, lidar_points.tolist())
                self.lidar_pub.publish(lidar_message)
            '''

            # Odometry
            odom = [
                np.array(self.env.robots[0].get_position()) - np.array(self.env.task.initial_pos),
                np.array(self.env.robots[0].get_rpy()) - np.array(self.env.task.initial_orn),
            ]

            self.br.sendTransform(
                (odom[0][0], odom[0][1], 0),
                tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1]),
                rospy.Time.now(),
                self.ns + "base_link",
                self.ns + "odom",
            )

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = self.ns + "odom"
            odom_msg.child_frame_id = self.ns + "base_link"

            odom_msg.pose.pose.position.x = odom[0][0]
            odom_msg.pose.pose.position.y = odom[0][1]
            (
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ) = tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1])

            odom_msg.twist.twist.linear.x = (cmdx + cmdy) * 5
            odom_msg.twist.twist.angular.z = (cmdy - cmdx) * 5 * 8.695652173913043
            self.odom_pub.publish(odom_msg)

            # Joint States
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.header.frame_id = ""
            #joint_state_msg.header.frame_id = self.ns + "odom"

            joint_names = self.env.robots[0].get_joint_names()

            joint_state_msg.name = joint_names
            joint_states_igibson = self.env.robots[0].get_joint_states()

            joint_state_msg.position = []
            joint_state_msg.velocity = []
            for jn in joint_names:
                jp = joint_states_igibson[jn][0]
                jv = joint_states_igibson[jn][1]
                #print(jn + ": " + str(jp) + ", " + str(jv))

                joint_state_msg.position.append(jp)
                joint_state_msg.velocity.append(jv)

            self.joint_states_pub.publish(joint_state_msg)

            #print("[SimNode::__init__] DEBUG INF")
            #while 1:
            #    continue

            '''
            # Ground truth pose
            gt_pose_msg = Odometry()
            gt_pose_msg.header.stamp = rospy.Time.now()
            gt_pose_msg.header.frame_id = self.ns + "odom"
            gt_pose_msg.child_frame_id = self.ns + "base_link"

            xyz = self.env.robots[0].get_position()
            rpy = self.env.robots[0].get_rpy()

            gt_pose_msg.pose.pose.position.x = xyz[0]
            gt_pose_msg.pose.pose.position.y = xyz[1]
            gt_pose_msg.pose.pose.position.z = xyz[2]
            (
                gt_pose_msg.pose.pose.orientation.x,
                gt_pose_msg.pose.pose.orientation.y,
                gt_pose_msg.pose.pose.orientation.z,
                gt_pose_msg.pose.pose.orientation.w,
            ) = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            gt_pose_msg.twist.twist.linear.x = cmdx
            gt_pose_msg.twist.twist.angular.z = -cmdy
            '''

    def cmd_callback(self, data):
        self.cmdx = data.linear.x
        self.cmdy = -data.angular.z
        self.last_update = rospy.Time.now()

    def tp_robot_callback(self, data):
        rospy.loginfo("Teleporting robot")
        position = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        orientation = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        ]
        self.env.robots[0].reset_new_pose(position, orientation)
        self.tp_time = rospy.Time.now()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ns = rospy.get_namespace()
    ns = ns[1:]

    print("============================================")
    print("[mobiman_jackal_jaco::__main__] ns: " + str(ns))
    print("============================================")

    node = SimNode(ns=ns)
    node.run()