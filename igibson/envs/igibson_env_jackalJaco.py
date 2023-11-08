import argparse
import logging
import os
import time
import math
import random
import yaml
import gym
import numpy as np
import pybullet as p
import pandas as pd
import gc
import pathlib
import pickle
from cv_bridge import CvBridge
from transforms3d.euler import euler2quat
from squaternion import Quaternion
from collections import OrderedDict

#from igibson import ros_path
from igibson.utils.utils import parse_config
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW

import rospkg
import rospy
import tf
import tf.transformations
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import MarkerArray, Marker

from ocs2_msgs.msg import collision_info # type: ignore 
from ocs2_msgs.srv import setDiscreteActionDRL, setContinuousActionDRL, setBool, setBoolResponse, setMPCActionResult, setMPCActionResultResponse # type: ignore

from drl.mobiman_drl_config import * # type: ignore 
from igibson.objects.ycb_object import YCBObject
from igibson.objects.ycb_object import StatefulObject
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from gazebo_msgs.msg import ModelStates

log = logging.getLogger(__name__)

'''
DESCRIPTION: iGibson Environment (OpenAI Gym interface).
'''
class iGibsonEnv(BaseEnv):

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 5.0,
        physics_timestep=1 / 60.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
        init_ros_node=False,
        ros_node_id=0,
        data_folder_path="",
        objects=None
    ):
        """
        ### NUA TODO: UPDATE!
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] START")

        ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
        ### Initialize Config Parameters
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] START CONFIG")
        config_igibson_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.config_igibson = parse_config(config_igibson_data)
        
        self.config_mobiman = Config(data_folder_path=data_folder_path) # type: ignore
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] END CONFIG")

        ### Initialize Variables
        self.init_flag = False
        self.init_goal_flag = False
        self.callback_update_flag = False
        self.episode_done = False
        self.reached_goal = False
        self.episode_num = 1
        self.step_num = 1
        self.total_step_num = 1
        self.total_collisions = 0
        self.total_rollover = 0
        self.total_goal = 0
        self.total_max_step = 0
        self.total_mpc_exit = 0
        self.total_target = 0
        self.total_time_horizon = 0
        self.total_last_step_distance = 0
        self.step_action = None
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.total_mean_episode_reward = 0.0
        #self.goal_status = Bool()
        #self.goal_status.data = False
        self.action_counter = 0
        self.observation_counter = 0
        self.mrt_ready = False
        self.mpc_action_result = 0
        self.mpc_action_complete = False

        # Variables for saving OARS data
        self.data = None
        self.oars_data = {'Index':[], 'Observation':[], 'Action':[], 'Reward':[]}
        self.idx = 1
        self.termination_reason = ''
        self.model_mode = -1
        
        self.init_robot_pose = {}
        self.robot_data = {}
        self.goal_data = {}
        self.target_data = {}
        self.arm_data = {}
        self.obs_data = {}
        self.target_msg = None

        self.training_data = []
        self.training_data.append(["episode_reward"])
        self.oar_data = []
        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        # Set initial command
        self.cmd_init_base = [0.0, 0.0]
        self.cmd_base = self.cmd_init_base
        self.cmd_init_j1 = 0.0
        self.cmd_init_j2 = 2.9
        self.cmd_init_j3 = 1.3
        self.cmd_init_j4 = 4.2
        self.cmd_init_j5 = 1.4
        self.cmd_init_j6 = 0.0
        self.cmd_init_arm = [self.cmd_init_j1, self.cmd_init_j2, self.cmd_init_j3, self.cmd_init_j4, self.cmd_init_j5, self.cmd_init_j6]
        self.cmd_arm = self.cmd_init_arm
        self.cmd = self.cmd_base + self.cmd_arm

        # Env objects
        self.objects = objects
        self.spawned_objects = []

        ## Set Observation-Action-Reward data filename
        self.oar_data_file = data_folder_path + "oar_data.csv"

        #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
        #while 1:
        #    continue
        
        ## Set namespace
        self.init_ros_node = init_ros_node
        self.ros_node_id = ros_node_id
        robot_ns = self.config_igibson["robot_ns"]
        self.ns = robot_ns + "_" + str(ros_node_id) + "/"

        #self.init_ros_env(self, ros_node_id=0, init_flag=True)

        '''
        if not self.ros_node_init:
            rospy.init_node("igibson_ros_" + str(ros_node_id), anonymous=True)

            self.listener = tf.TransformListener()

            # ROS variables
            self.last_update_base = rospy.Time.now()
            self.last_update_arm = rospy.Time.now()
            self.bridge = CvBridge()
            self.br = tf.TransformBroadcaster()

            # Subscribers
            rospy.Subscriber(self.ns + self.config_mobiman.base_control_msg_name, Twist, self.cmd_base_callback)
            rospy.Subscriber(self.ns + self.config_mobiman.arm_control_msg_name, JointTrajectory, self.cmd_arm_callback)

            rospy.Subscriber(self.ns + self.config_mobiman.target_msg_name, MarkerArray, self.callback_target)
            rospy.Subscriber(self.ns + self.config_mobiman.occgrid_msg_name, OccupancyGrid, self.callback_occgrid)
            rospy.Subscriber(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info, self.callback_selfcoldistance)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info, self.callback_extcoldistance_base)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info, self.callback_extcoldistance_arm) # type: ignore
            rospy.Subscriber(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray, self.callback_pointsonrobot)

            # Publishers
            self.image_pub = rospy.Publisher(self.ns + self.config_mobiman.rgb_image_msg_name, Image, queue_size=10)
            self.depth_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_msg_name, Image, queue_size=10)
            self.depth_raw_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_raw_msg_name, Image, queue_size=10)
            self.camera_info_pub = rospy.Publisher(self.ns + self.config_mobiman.camera_info_msg_name, CameraInfo, queue_size=10)
            self.lidar_pub = rospy.Publisher(self.ns + self.config_mobiman.lidar_msg_name, PointCloud2, queue_size=10)
            self.odom_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.odom_gt_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.joint_states_pub = rospy.Publisher(self.ns + self.config_mobiman.arm_state_msg_name, JointState, queue_size=10)
            #self.goal_status_pub = rospy.Publisher(self.config_mobiman.goal_status_msg_name, Bool, queue_size=1)
            #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
            self.debug_visu_pub = rospy.Publisher(self.ns + 'debug_visu', MarkerArray, queue_size=1)
            self.model_state_pub = rospy.Publisher(self.ns + "model_states", ModelStates, queue_size=10)

            # Clients

            # Services
            rospy.Service(self.ns + 'set_mrt_ready', setBool, self.service_set_mrt_ready)
            rospy.Service(self.ns + 'set_mpc_action_result', setMPCActionResult, self.service_set_mpc_action_result)

            #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
            #while 1:
            #    continue
        '''

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] BEFORE super")
        super(iGibsonEnv, self).__init__(
              config_file=config_file,
              scene_id=scene_id,
              mode=mode,
              action_timestep=action_timestep,
              physics_timestep=physics_timestep,
              rendering_settings=rendering_settings,
              vr_settings=vr_settings,
              device_idx=device_idx,
              use_pb_gui=use_pb_gui,
        )
        self.automatic_reset = automatic_reset
        self.init_flag = True

        '''
        # Timers
        self.create_objects(self.objects)
        self.transform_timer = rospy.Timer(rospy.Duration(0.01), self.timer_transform)
        
        self.timer = rospy.Timer(rospy.Duration(0.05), self.callback_update) # type: ignore

        # Wait for topics
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.selfcoldistance_msg_name) + "...")
        rospy.wait_for_message(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info)

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.extcoldistance_base_msg_name) + "...")
        rospy.wait_for_message(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info)

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.callback_extcoldistance_arm) + "...")
        rospy.wait_for_message(self.ns + self.config_mobiman.callback_extcoldistance_arm, collision_info)

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.pointsonrobot_msg_name) + "...")
        rospy.wait_for_message(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray)

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting callback_update_flag...")
        while not self.callback_update_flag:
            continue
        
        self.init_flag = True
        '''

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] num robots: " + str(len(self.robots)))
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] END")

        #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG INF")
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def init_ros_env(self, ros_node_id=0, init_flag=True):
        print("[igibson_env_jackalJaco::iGibsonEnv::init_ros_env] START")
        if init_flag:
            print("[igibson_env_jackalJaco::iGibsonEnv::init_ros_env] ROS entered to the chat!")
            rospy.init_node("igibson_ros_" + str(ros_node_id), anonymous=True)

            self.listener = tf.TransformListener()

            # ROS variables
            self.last_update_base = rospy.Time.now()
            self.last_update_arm = rospy.Time.now()
            self.bridge = CvBridge()
            self.br = tf.TransformBroadcaster()

            # Subscribers
            rospy.Subscriber(self.ns + self.config_mobiman.base_control_msg_name, Twist, self.cmd_base_callback)
            rospy.Subscriber(self.ns + self.config_mobiman.arm_control_msg_name, JointTrajectory, self.cmd_arm_callback)

            #rospy.Subscriber(self.ns + self.config_mobiman.target_msg_name, MarkerArray, self.callback_target)
            rospy.Subscriber(self.ns + self.config_mobiman.occgrid_msg_name, OccupancyGrid, self.callback_occgrid)
            rospy.Subscriber(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info, self.callback_selfcoldistance)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info, self.callback_extcoldistance_base)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info, self.callback_extcoldistance_arm) # type: ignore
            rospy.Subscriber(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray, self.callback_pointsonrobot)

            # Publishers
            self.image_pub = rospy.Publisher(self.ns + self.config_mobiman.rgb_image_msg_name, Image, queue_size=10)
            self.depth_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_msg_name, Image, queue_size=10)
            self.depth_raw_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_raw_msg_name, Image, queue_size=10)
            self.camera_info_pub = rospy.Publisher(self.ns + self.config_mobiman.camera_info_msg_name, CameraInfo, queue_size=10)
            self.lidar_pub = rospy.Publisher(self.ns + self.config_mobiman.lidar_msg_name, PointCloud2, queue_size=10)
            self.odom_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.odom_gt_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.joint_states_pub = rospy.Publisher(self.ns + self.config_mobiman.arm_state_msg_name, JointState, queue_size=10)
            #self.goal_status_pub = rospy.Publisher(self.config_mobiman.goal_status_msg_name, Bool, queue_size=1)
            #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
            self.debug_visu_pub = rospy.Publisher(self.ns + 'debug_visu', MarkerArray, queue_size=1)
            self.model_state_pub = rospy.Publisher(self.ns + "model_states", ModelStates, queue_size=10)

            # Clients

            # Services
            rospy.Service(self.ns + 'set_mrt_ready', setBool, self.service_set_mrt_ready)
            rospy.Service(self.ns + 'set_mpc_action_result', setMPCActionResult, self.service_set_mpc_action_result)

            # Timers
            self.create_objects(self.objects)
            self.transform_timer = rospy.Timer(rospy.Duration(0.01), self.timer_transform)
            
            self.timer = rospy.Timer(rospy.Duration(0.05), self.callback_update) # type: ignore

            # Wait for topics
            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.selfcoldistance_msg_name) + "...")
            rospy.wait_for_message(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info)

            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.extcoldistance_base_msg_name) + "...")
            rospy.wait_for_message(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info)

            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.extcoldistance_arm_msg_name) + "...")
            rospy.wait_for_message(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info)

            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.pointsonrobot_msg_name) + "...")
            rospy.wait_for_message(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray)

            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting callback_update_flag...")
            while not self.callback_update_flag:
                continue

            #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
            #while 1:
            #    continue
        print("[igibson_env_jackalJaco::iGibsonEnv::init_ros_env] END")

    '''
    DESCRIPTION: TODO...
    '''
    def create_objects(self, objects):
        for key,val in objects.items():
            pointer = YCBObject(name=val, abilities={"soakable": {}, "cleaningTool": {}})
            self.simulator.import_object(pointer)
            self.spawned_objects.append(pointer)
            self.spawned_objects[-1].set_position([3,3,0.2])
            self.spawned_objects[-1].set_orientation([0.7071068, 0, 0, 0.7071068])

    '''
    DESCRIPTION: TODO...
    '''
    def timer_transform(self, timer):
        #print("[igibson_env_jackalJaco::iGibsonEnv::timer_transform] START")
        model_state_msg = ModelStates()
        pose = Pose()
        for obj, dict in zip(self.spawned_objects, self.objects.items()):
            # self.br.sendTransform(obj.get_position(), obj.get_orientation(), rospy.Time.now(), f'{self.ns}{dict[0]}', 'world')
            model_state_msg.name.append(dict[0])
            x,y,z = obj.get_position()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            x,y,z,w = obj.get_orientation()
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            pose.orientation.w = w
            model_state_msg.pose.append(pose)
        self.model_state_pub.publish(model_state_msg)
        #print("[igibson_env_jackalJaco::iGibsonEnv::timer_transform] END")

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_base_callback(self, data):
        self.cmd_base = [data.linear.x, data.angular.z]
        self.last_update_base = rospy.Time.now()

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_arm_callback(self, data):
        joint_names = data.joint_names
        self.cmd_arm = list(data.points[0].positions)

    '''
    DESCRIPTION: TODO...
    '''
    def callback_target(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_target] INCOMING")
        self.target_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_occgrid(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_occgrid] INCOMING")
        self.occgrid_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_selfcoldistance(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_selfcoldistance] INCOMING")
        self.selfcoldistance_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_base(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_base] INCOMING")
        self.extcoldistance_base_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_arm(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_arm] INCOMING")
        self.extcoldistance_arm_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_pointsonrobot(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_pointsonrobot] INCOMING")
        self.pointsonrobot_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_update(self, event):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] START")

        goal_frame_name = self.ns + self.config_mobiman.goal_frame_name
        #ee_frame_name = self.ns + self.config_mobiman.ee_frame_name
        #robot_frame_name = self.ns + self.config_mobiman.robot_frame_name

        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] world_frame_name: " + str(self.config_mobiman.world_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] goal_frame_name: " + str(goal_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] ee_frame_name: " + str(ee_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] robot_frame_name: " + str(robot_frame_name))

        self.update_robot_data()
        self.update_arm_data()
        self.update_ros_topics()

        '''
        try:
            self.listener.waitForTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=self.config_mobiman.robot_frame_name, time=rospy.Time(0), timeout=rospy.Duration(1))
            (self.trans_robot_wrt_world, self.rot_robot_wrt_world) = self.listener.lookupTransform(self.config_mobiman.world_frame_name, self.config_mobiman.robot_frame_name, rospy.Time(0))
            
        except Exception as e0:
            #print(e0)
            ...
        
        try:
            self.listener.waitForTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=self.config_mobiman.ee_frame_name, time=rospy.Time(0), timeout=rospy.Duration(1))
            (self.trans_ee_wrt_world, self.rot_ee_wrt_world) = self.listener.lookupTransform(self.config_mobiman.world_frame_name, self.config_mobiman.ee_frame_name, rospy.Time(0))
            
        except Exception as e1:
            #print(e1) 
            ...           
        '''

        #a = False
        #b = False
        #c = False

        try:
            (self.trans_goal_wrt_world, self.rot_goal_wrt_world) = self.listener.lookupTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=goal_frame_name, time=rospy.Time(0))

            trans_robot_wrt_world = tf.transformations.translation_matrix([self.robot_data["x"], self.robot_data["y"], self.robot_data["z"]])
            quat_robot_wrt_world = tf.transformations.quaternion_matrix([self.robot_data["qx"], self.robot_data["qy"], self.robot_data["qz"], self.robot_data["qw"]])
            tf_robot_wrt_world = tf.transformations.concatenate_matrices(trans_robot_wrt_world, quat_robot_wrt_world)

            #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED trans_robot_wrt_world")
            #print(trans_robot_wrt_world)
            #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED tf_robot_wrt_world")
            #print(tf_robot_wrt_world)
            #print("------")

            trans_ee_wrt_world = tf.transformations.translation_matrix([self.arm_data["x"], self.arm_data["y"], self.arm_data["z"]])
            quat_ee_wrt_world = tf.transformations.quaternion_matrix([self.arm_data["qx"], self.arm_data["qy"], self.arm_data["qz"], self.arm_data["qw"]])
            tf_ee_wrt_world = tf.transformations.concatenate_matrices(trans_ee_wrt_world, quat_ee_wrt_world)
            
            trans_goal_wrt_world = tf.transformations.translation_matrix(self.trans_goal_wrt_world)
            quat_goal_wrt_world = tf.transformations.quaternion_matrix(self.rot_goal_wrt_world)
            tf_goal_wrt_world = tf.transformations.concatenate_matrices(trans_goal_wrt_world, quat_goal_wrt_world)

            #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED trans_goal_wrt_world")
            #print(trans_goal_wrt_world)
            #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED tf_goal_wrt_world")
            #print(tf_goal_wrt_world)
            #print("------")

            # Calculate the transformation from end effector wrt base
            tf_world_wrt_robot = tf.transformations.inverse_matrix(tf_robot_wrt_world)
            transform_goal_wrt_robot = tf.transformations.concatenate_matrices(tf_world_wrt_robot, tf_goal_wrt_world)
            
            tf_world_wrt_ee = tf.transformations.inverse_matrix(tf_ee_wrt_world)
            transform_goal_wrt_ee = tf.transformations.concatenate_matrices(tf_world_wrt_ee, tf_goal_wrt_world)
            
            '''
            print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED tf_robot_wrt_world")
            print(tf_robot_wrt_world)
            print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED tf_world_wrt_robot")
            print(tf_world_wrt_robot)
            print("------")
            print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED transform_goal_wrt_robot")
            print(transform_goal_wrt_robot)
            print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] CALCULATED transform_goal_wrt_ee")
            print(transform_goal_wrt_ee)
            print("------")
            '''

            self.trans_goal_wrt_robot = tf.transformations.translation_from_matrix(transform_goal_wrt_robot)
            self.rot_goal_wrt_robot = tf.transformations.quaternion_from_matrix(transform_goal_wrt_robot)
            
            self.trans_goal_wrt_ee = tf.transformations.translation_from_matrix(transform_goal_wrt_ee)
            self.rot_goal_wrt_ee = tf.transformations.quaternion_from_matrix(transform_goal_wrt_ee)
            
            self.callback_update_flag = True

            #a = True
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAa")
            self.update_goal_data()
            self.update_goal_data_wrt_robot()
            self.update_goal_data_wrt_ee()
        except Exception as e2:
            #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] " + str(e2))
            ...

        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] END")

    '''
    DESCRIPTION: TODO...
    '''
    def client_movebase(self):
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = 0.5
        goal.target_pose.pose.orientation.w = 1.0

        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return client.get_result()

    '''
    DESCRIPTION: TODO...
    '''
    def client_set_action_drl(self, action, last_step_flag=False):
        #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Waiting for service set_action_drl...")
        set_action_drl_service_name = self.ns + 'set_action_drl'
        #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] set_action_drl_service_name: " + str(set_action_drl_service_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] last_step_flag: " + str(last_step_flag))
        rospy.wait_for_service(set_action_drl_service_name)
        #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Received service set_action_drl!")
        try:
            if self.config_mobiman.action_type == 0:
                #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] DISCRETE ACTION")
                srv_set_discrete_action_drl = rospy.ServiceProxy(set_action_drl_service_name, setDiscreteActionDRL)            
                success = srv_set_discrete_action_drl(action, self.config_mobiman.action_time_horizon).success
            else:
                #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] CONTINUOUS ACTION")
                srv_set_continuous_action_drl = rospy.ServiceProxy(set_action_drl_service_name, setContinuousActionDRL)
                success = srv_set_continuous_action_drl(action, self.config_mobiman.action_time_horizon, last_step_flag, self.config_mobiman.last_step_distance_threshold).success
                #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] CONTINUOUS ACTION SENT!")
            '''
            if(success):
                print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Updated action: " + str(action))
            else:
                #print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] goal_pose is NOT updated!")
                print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] ERROR: set_action_drl is NOT successful!")
            '''
            return success

        except rospy.ServiceException as e:  
            print("[igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] ERROR: Service call failed: %s"%e)
            return False

    '''
    DESCRIPTION: TODO...
    '''
    def service_set_mrt_ready(self, req):
        self.mrt_ready = req.val

        #print("[igibson_env_jackalJaco::iGibsonEnv::service_set_mrt_ready] DEBUG_INF")
        #while 1:
        #    continue

        return setBoolResponse(True)

    '''
    DESCRIPTION: TODO...
    '''
    def service_set_mpc_action_result(self, req):
        # 0: MPC/MRT Failure
        # 1: Collision
        # 2: Rollover
        # 3: Goal reached
        # 4: Target reached
        # 5: Time-horizon reached
        #print("[igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] START")
        self.mpc_action_result = req.action_result
            
        if self.mpc_action_result == 0:
            self.termination_reason = 'mpc_exit'
            self.total_mpc_exit += 1
        elif self.mpc_action_result == 1:
            self.termination_reason = 'collision'
            self.total_collisions += 1
            self.episode_done = True
        elif self.mpc_action_result == 2:
            self.termination_reason = 'rollover'
            self.total_rollover += 1
            self.episode_done = True
        elif self.mpc_action_result == 3:
            self.termination_reason = 'goal'
            self.total_goal += 1
            self.reached_goal = True
            self.episode_done = True
        elif self.mpc_action_result == 4:
            self.termination_reason = 'target'
            self.total_target += 1
        elif self.mpc_action_result == 5:
            self.termination_reason = 'time_horizon'
            self.total_time_horizon += 1
        
        self.model_mode = req.model_mode
        self.mpc_action_complete = True
        #print("[igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] mpc_action_result: " + str(self.mpc_action_result) + " " + str(self.ns))
        #print("[igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] END")
        return setMPCActionResultResponse(True)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_selfcoldistance_config(self):
        n_selfcoldistance = int(len(self.selfcoldistance_msg.distance))
        self.config_mobiman.set_selfcoldistance_config(n_selfcoldistance)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_extcoldistance_base_config(self):
        n_extcoldistance_base = int(len(self.extcoldistance_base_msg.distance))
        self.config_mobiman.set_extcoldistance_base_config(n_extcoldistance_base)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_extcoldistance_arm_config(self):
        n_extcoldistance_arm = int(len(self.extcoldistance_arm_msg.distance))
        self.config_mobiman.set_extcoldistance_arm_config(n_extcoldistance_arm)

    '''
    DESCRIPTION: TODO... Update robot data
    '''
    def update_robot_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] START " )

        robot_xyz, robot_quat = self.robots[0].get_position_orientation()
        robot_rpy = self.robots[0].get_rpy()

        self.robot_data["x"] = robot_xyz[0] # type: ignore
        self.robot_data["y"] = robot_xyz[1] # type: ignore
        self.robot_data["z"] = robot_xyz[2] # type: ignore
        
        self.robot_data["qx"] = robot_quat[0] # type: ignore
        self.robot_data["qy"] = robot_quat[1] # type: ignore
        self.robot_data["qz"] = robot_quat[2] # type: ignore
        self.robot_data["qw"] = robot_quat[3] # type: ignore

        self.robot_data["roll"] = robot_rpy[0]
        self.robot_data["pitch"] = robot_rpy[1]
        self.robot_data["yaw"] = robot_rpy[2]

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] x: " + str(self.robot_data["x"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] y: " + str(self.robot_data["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] z: " + str(self.robot_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qx: " + str(self.robot_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qy: " + str(self.robot_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qz: " + str(self.robot_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qw: " + str(self.robot_data["qw"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] END" )

    '''
    DESCRIPTION: TODO... Update arm data
    '''
    def update_arm_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] START " )
        
        #link_names = self.robots[0].get_link_names()
        #base_pos, base_quat = self.robots[0].get_base_link_position_orientation()
        ee_pos, ee_quat = self.robots[0].get_link_position_orientation(self.config_mobiman.ee_frame_name)
        ee_rpy = self.robots[0].get_link_rpy(self.config_mobiman.ee_frame_name)
        
        '''
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] link_names: " )
        print(link_names)
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_frame_name: " + str(self.config_mobiman.ee_frame_name))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_pos: " + str(ee_pos))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_quat: " + str(ee_quat))

        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_pos: " + str(base_pos))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_quat: " + str(base_quat))

        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] DEBUG_INF" )
        while 1:
            continue
        '''

        self.arm_data["x"] = ee_pos[0] # type: ignore
        self.arm_data["y"] = ee_pos[1] # type: ignore
        self.arm_data["z"] = ee_pos[2] # type: ignore
        self.arm_data["qx"] = ee_quat[0] # type: ignore
        self.arm_data["qy"] = ee_quat[1] # type: ignore
        self.arm_data["qz"] = ee_quat[2] # type: ignore
        self.arm_data["qw"] = ee_quat[3] # type: ignore
        
        #q = Quaternion(rot[3], rot[0], rot[1], rot[2]) # type: ignore
        #e = q.to_euler(degrees=False)
        self.arm_data["roll"] = ee_rpy[0]
        self.arm_data["pitch"] = ee_rpy[1]
        self.arm_data["yaw"] = ee_rpy[2]

        ## Joint positions and velocities
        joint_names = self.config_mobiman.arm_joint_names
        joint_states_igibson = self.robots[0].get_joint_states()

        self.arm_data["joint_name"] = joint_names
        self.arm_data["joint_pos"] = []
        self.arm_data["joint_velo"] = []
        for jn in joint_names:
            jp = joint_states_igibson[jn][0]
            jv = joint_states_igibson[jn][1]
            #print(jn + ": " + str(jp) + ", " + str(jv))

            # Normalize yaw difference to be within range of -pi to pi
            while jp > math.pi:
                jp -= 2*math.pi
            while jp < -math.pi:
                jp += 2*math.pi
            
            self.arm_data["joint_pos"].append(jp)
            self.arm_data["joint_velo"].append(jv)

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] x: " + str(self.arm_data["x"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] y: " + str(self.arm_data["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] z: " + str(self.arm_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qx: " + str(self.arm_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qy: " + str(self.arm_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qz: " + str(self.arm_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qw: " + str(self.arm_data["qw"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] joint_names: ")
        #print(joint_names)
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] joint_pos: ")
        #print(self.arm_data["joint_pos"])
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] END" )

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] DEBUG_INF" )
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] START")
        
        translation_wrt_world = self.trans_goal_wrt_world
        rotation_wrt_world = self.rot_goal_wrt_world
        
        self.goal_data["x"] = translation_wrt_world[0] # type: ignore
        self.goal_data["y"] = translation_wrt_world[1] # type: ignore
        self.goal_data["z"] = translation_wrt_world[2] # type: ignore
        self.goal_data["qx"] = rotation_wrt_world[0] # type: ignore
        self.goal_data["qy"] = rotation_wrt_world[1] # type: ignore
        self.goal_data["qz"] = rotation_wrt_world[2] # type: ignore
        self.goal_data["qw"] = rotation_wrt_world[3] # type: ignore

        q = Quaternion(rotation_wrt_world[3], rotation_wrt_world[0], rotation_wrt_world[1], rotation_wrt_world[2]) # type: ignore
        e = q.to_euler(degrees=False)
        self.goal_data["roll"] = e[0] # type: ignore
        self.goal_data["pitch"] = e[1] # type: ignore
        self.goal_data["yaw"] = e[2] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] x: " + str(self.goal_data["x"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] y: " + str(self.goal_data["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] z: " + str(self.goal_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qx: " + str(self.goal_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qy: " + str(self.goal_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qz: " + str(self.goal_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qw: " + str(self.goal_data["qw"]))

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_robot(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] START")

        translation_wrt_robot = self.trans_goal_wrt_robot
        rotation_wrt_robot = self.rot_goal_wrt_robot

        self.goal_data["x_wrt_robot"] = translation_wrt_robot[0] # type: ignore
        self.goal_data["y_wrt_robot"] = translation_wrt_robot[1] # type: ignore
        self.goal_data["z_wrt_robot"] = translation_wrt_robot[2] # type: ignore
        self.goal_data["qx_wrt_robot"] = rotation_wrt_robot[0] # type: ignore
        self.goal_data["qy_wrt_robot"] = rotation_wrt_robot[1] # type: ignore
        self.goal_data["qz_wrt_robot"] = rotation_wrt_robot[2] # type: ignore
        self.goal_data["qw_wrt_robot"] = rotation_wrt_robot[3] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] x_wrt_robot: " + str(self.goal_data["x_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] y_wrt_robot: " + str(self.goal_data["y_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] z_wrt_robot: " + str(self.goal_data["z_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qx_wrt_robot: " + str(self.goal_data["qx_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qy_wrt_robot: " + str(self.goal_data["qy_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qz_wrt_robot: " + str(self.goal_data["qz_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qw_wrt_robot: " + str(self.goal_data["qw_wrt_robot"]))

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_ee(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] START")

        translation_wrt_ee = self.trans_goal_wrt_ee
        rotation_wrt_ee = self.rot_goal_wrt_ee

        self.goal_data["x_wrt_ee"] = translation_wrt_ee[0] # type: ignore
        self.goal_data["y_wrt_ee"] = translation_wrt_ee[1] # type: ignore
        self.goal_data["z_wrt_ee"] = translation_wrt_ee[2] # type: ignore
        self.goal_data["qx_wrt_ee"] = rotation_wrt_ee[0] # type: ignore
        self.goal_data["qy_wrt_ee"] = rotation_wrt_ee[1] # type: ignore
        self.goal_data["qz_wrt_ee"] = rotation_wrt_ee[2] # type: ignore
        self.goal_data["qw_wrt_ee"] = rotation_wrt_ee[3] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] x_wrt_ee: " + str(self.goal_data["x_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] y_wrt_ee: " + str(self.goal_data["y_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] z_wrt_ee: " + str(self.goal_data["z_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qx_wrt_ee: " + str(self.goal_data["qx_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qy_wrt_ee: " + str(self.goal_data["qy_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qz_wrt_ee: " + str(self.goal_data["qz_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qw_wrt_ee: " + str(self.goal_data["qw_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_target_data(self, x, y, z, roll, pitch, yaw):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] START")
        
        ## NUA TODO: Generalize to multiple target points!
        self.target_data["x"] = x
        self.target_data["y"] = y
        self.target_data["z"] = z
        
        q = Quaternion() # type: ignore
        q = q.from_euler(roll, pitch, yaw)
        self.target_data["qx"] = q.x
        self.target_data["qy"] = q.y
        self.target_data["qz"] = q.z
        self.target_data["qw"] = q.w

        self.target_data["roll"] = roll # type: ignore
        self.target_data["pitch"] = pitch # type: ignore
        self.target_data["yaw"] = pitch # type: ignore

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] UPDATED.")

        '''
        p = Point()
        p.x = self.target_data["x"]
        p.y = self.target_data["y"]
        p.z = self.target_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_ros_topics(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] START")

        now = rospy.Time.now()
        if (now - self.last_update_base).to_sec() > 0.1:
            self.cmd_base = [0.0, 0.0]

        ## Odom
        odom = [
            np.array(self.robots[0].get_position()),
            np.array(self.robots[0].get_rpy()),
        ]

        self.br.sendTransform(
            (odom[0][0], odom[0][1], 0),
            tf.transformations.quaternion_from_euler(odom[-1][0], odom[-1][1], odom[-1][2]), # type: ignore
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
        ) = tf.transformations.quaternion_from_euler(odom[-1][0], odom[-1][1], odom[-1][2]) # type: ignore

        odom_msg.twist.twist.linear.x = self.robots[0].get_linear_velocity()[0]
        odom_msg.twist.twist.linear.y = self.robots[0].get_linear_velocity()[1]
        odom_msg.twist.twist.linear.z = self.robots[0].get_linear_velocity()[2]
        odom_msg.twist.twist.angular.x = self.robots[0].get_angular_velocity()[0]
        odom_msg.twist.twist.angular.y = self.robots[0].get_angular_velocity()[1]
        odom_msg.twist.twist.angular.z = self.robots[0].get_angular_velocity()[2]
        self.odom_pub.publish(odom_msg)
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics2] odom_msg: " + str(odom_msg))

        # Joint States
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.header.frame_id = ""
        #joint_state_msg.header.frame_id = self.ns + "odom"

        joint_names = self.robots[0].get_joint_names()

        joint_state_msg.name = joint_names
        joint_states_igibson = self.robots[0].get_joint_states()

        joint_state_msg.position = []
        joint_state_msg.velocity = []
        for jn in joint_names:
            jp = joint_states_igibson[jn][0]
            jv = joint_states_igibson[jn][1]
            #print(jn + ": " + str(jp) + ", " + str(jv))
            joint_state_msg.position.append(jp)
            joint_state_msg.velocity.append(jv)

        self.joint_states_pub.publish(joint_state_msg)

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_observation(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] START")
        if self.config_mobiman.observation_space_type == "mobiman_FC":
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] mobiman_FC")

            # Get OccGrid array observation
            #obs_occgrid = self.get_obs_occgrid()

            # Get collision sphere distance observation
            obs_selfcoldistance = self.get_obs_selfcoldistance()
            obs_extcoldistance_base = self.get_obs_extcoldistance_base()
            obs_extcoldistance_arm = self.get_obs_extcoldistance_arm()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update arm joint observation
            obs_armstate = self.get_obs_armstate()

            # Update observation data
            #self.obs_data["occgrid"] = np.vstack((self.obs_data["occgrid"], obs_occgrid))
            #self.obs_data["occgrid"] = np.delete(self.obs_data["occgrid"], np.s_[0], axis=0)

            self.obs_data["selfcoldistance"] = np.vstack((self.obs_data["selfcoldistance"], obs_selfcoldistance))
            self.obs_data["selfcoldistance"] = np.delete(self.obs_data["selfcoldistance"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_base"] = np.vstack((self.obs_data["extcoldistance_base"], obs_extcoldistance_base))
            self.obs_data["extcoldistance_base"] = np.delete(self.obs_data["extcoldistance_base"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_arm"] = np.vstack((self.obs_data["extcoldistance_arm"], obs_extcoldistance_arm))
            self.obs_data["extcoldistance_arm"] = np.delete(self.obs_data["extcoldistance_arm"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            self.obs_data["armstate"] = np.vstack((self.obs_data["armstate"], obs_armstate))
            self.obs_data["armstate"] = np.delete(self.obs_data["armstate"], np.s_[0], axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data selfcoldistance shape: " + str(self.obs_data["selfcoldistance"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistance_base shape: " + str(self.obs_data["extcoldistance_base"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistance_arm shape: " + str(self.obs_data["extcoldistance_arm"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data armstate shape: " + str(self.obs_data["armstate"].shape))

            # Update observation
            '''
            obs_stacked_occgrid = self.obs_data["occgrid"][-1,:].reshape(self.config_mobiman.fc_obs_shape)

            if self.config_mobiman.n_obs_stack > 1: # type: ignore
                latest_index = (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack) - 1 # type: ignore
                j = 0
                for i in range(latest_index-1, -1, -1): # type: ignore
                    j += 1
                    if j % self.config_mobiman.n_skip_obs_stack == 0: # type: ignore
                        obs_stacked_occgrid = np.hstack((self.obs_data["occgrid"][i,:], obs_stacked_occgrid))
            '''

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_stacked_occgrid shape: " + str(obs_stacked_occgrid.shape))

            #self.obs = np.concatenate((obs_stacked_occgrid, obs_extcoldistancedist, obs_goal), axis=0)
            self.obs = np.concatenate((obs_selfcoldistance, obs_extcoldistance_base, obs_extcoldistance_arm, obs_goal, obs_armstate), axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs: " + str(self.obs.shape))

        elif self.config_mobiman.observation_space_type == "mobiman_2DCNN_FC":

            print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] NEEDS REVIEW: DEBUG INF")
            while 1:
                continue

            # Get OccGrid image observation
            obs_occgrid_image = self.get_obs_occgrid(image_flag=True)
            obs_occgrid_image = np.expand_dims(obs_occgrid_image, axis=0)

            # Get collision sphere distance observation
            obs_selfcoldistance = self.get_obs_selfcoldistance()
            obs_extcoldistance_base = self.get_obs_extcoldistance_base()
            obs_extcoldistance_arm = self.get_obs_extcoldistance_arm()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update observation data
            self.obs_data["occgrid_image"] = np.vstack((self.obs_data["occgrid_image"], obs_occgrid_image))
            self.obs_data["occgrid_image"] = np.delete(self.obs_data["occgrid_image"], np.s_[0], axis=0)

            self.obs_data["selfcoldistance"] = np.vstack((self.obs_data["selfcoldistance"], obs_selfcoldistance))
            self.obs_data["selfcoldistance"] = np.delete(self.obs_data["selfcoldistance"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_base"] = np.vstack((self.obs_data["extcoldistance_base"], obs_extcoldistance_base))
            self.obs_data["extcoldistance_base"] = np.delete(self.obs_data["extcoldistance_base"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_arm"] = np.vstack((self.obs_data["extcoldistance_arm"], obs_extcoldistance_arm))
            self.obs_data["extcoldistance_arm"] = np.delete(self.obs_data["extcoldistance_arm"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            # Update observation
            obs_space_occgrid_image = self.obs_data["occgrid_image"][-1,:,:]
            obs_space_occgrid_image = np.expand_dims(obs_space_occgrid_image, axis=0)

            if self.config_mobiman.n_obs_stack > 1: # type: ignore
                if(self.config_mobiman.n_skip_obs_stack > 1): # type: ignore
                    latest_index = (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack) - 1 # type: ignore
                    j = 0
                    for i in range(latest_index-1, -1, -1): # type: ignore
                        j += 1
                        if j % self.config_mobiman.n_skip_obs_stack == 0: # type: ignore

                            obs_space_occgrid_image_current = self.obs_data["occgrid_image"][i,:,:]
                            obs_space_occgrid_image_current = np.expand_dims(obs_space_occgrid_image_current, axis=0)
                            obs_space_occgrid_image = np.vstack([obs_space_occgrid_image_current, obs_space_occgrid_image])
                
                else:
                    obs_space_occgrid_image = self.obs_data["occgrid_image"]

            obs_space_coldistance_goal = np.concatenate((obs_selfcoldistance, obs_extcoldistance_base, obs_extcoldistance_arm, obs_goal), axis=0)

            #print("**************** " + str(self.step_num))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data occgrid_image shape: " + str(self.obs_data["occgrid_image"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistancedist shape: " + str(self.obs_data["extcoldistancedist"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image: ")
            ##print(obs_space_laser_image[0, 65:75])
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target dist: " + str(obs_target[0,0]))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target angle: " + str(obs_target[0,1] * 180 / math.pi))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] previous_action: " + str(self.previous_action))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_occgrid_image shape: " + str(obs_occgrid_image.shape))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image type: " + str(type(obs_space_laser_image)))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_occgrid_image shape: " + str(obs_space_occgrid_image.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_coldistance_goal shape: " + str(obs_space_coldistance_goal.shape))
            #print("****************")

            self.obs["occgrid_image"] = obs_space_occgrid_image
            self.obs["coldistance_goal"] = obs_space_coldistance_goal
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] END")

    '''
    DESCRIPTION: TODO...
    '''
    def take_action(self, action):
        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] START")
        
        self.step_action = action
        self.current_step += 1
        
        #action = [0.0, 2.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]

        self.update_target_data(action[2], action[3], action[4], action[5], action[6], action[7])

        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] total_step_num: " + str(self.total_step_num))
        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] action: " + str(action))
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting for mrt_ready...")
        while not self.mrt_ready:
            continue
        self.mrt_ready = False
        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Recieved mrt_ready!")

        '''
        # Run Action Server
        success = self.client_set_action_drl(action)
        while 1:
            cmd = self.cmd_base + self.cmd_arm
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] " + self.ns + " cmd: " + str(cmd))

            self.robots[0].apply_action(cmd)
            collision_links = self.run_simulation()
        '''

        if self.config_mobiman.ablation_mode == 0:
            # Run Action Server
            success = self.client_set_action_drl(action)

            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting mpc_action_complete for " + str(self.config_mobiman.action_time_horizon) + " sec... " + str(self.ns))
            #rospy.sleep(self.config_mobiman.action_time_horizon)
            time_start = time.time()
            while not self.mpc_action_complete:
                cmd = self.cmd_base + self.cmd_arm
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] cmd: " + str(cmd))
                self.robots[0].apply_action(cmd)
                self.simulator_step()
                #collision_links = self.run_simulation()
                #self.collision_links = collision_links
                #self.collision_step += int(len(collision_links) > 0)

            time_end = time.time()
            self.dt_action = time_end - time_start
            self.mpc_action_complete = False
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Action completed in " + str(self.dt_action) + " sec! " + str(self.ns))
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] mpc_action_result: " + str(self.mpc_action_result))

            distance2goal = self.get_base_distance2goal_2D()
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] distance2goal: " + str(distance2goal))
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] last_step_distance_threshold: " + str(self.config_mobiman.last_step_distance_threshold))
            
            if distance2goal < self.config_mobiman.last_step_distance_threshold: # type: ignore
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] WITHIN THE DIST, SETTING TARGET TO GOAL!" )

                self.update_target_data(self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"])
                last_action = [1.0, 1.0, self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"]]
                success = self.client_set_action_drl(last_action, True)

                self.total_last_step_distance += 1
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting LAST mpc_action_complete for " + str(self.config_mobiman.action_time_horizon) + " sec... " + str(self.ns))
                #rospy.sleep(self.config_mobiman.action_time_horizon)
                time_start = time.time()
                while not self.mpc_action_complete:
                    cmd = self.cmd_base + self.cmd_arm
                    #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] cmd: " + str(cmd))
                    self.robots[0].apply_action(cmd)
                    self.simulator_step()
                    #collision_links = self.run_simulation()
                    #self.collision_links = collision_links
                    #self.collision_step += int(len(collision_links) > 0)

                time_end = time.time()
                self.dt_action += time_end - time_start
                self.mpc_action_complete = False
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] LAST Action completed in " + str(self.dt_action) + " sec! " + str(self.ns))
        
        elif self.config_mobiman.ablation_mode == 1:
            distance2goal = self.get_base_distance2goal_2D()
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] distance2goal: " + str(distance2goal))
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] last_step_distance_threshold: " + str(self.config_mobiman.last_step_distance_threshold))
            
            if distance2goal < self.config_mobiman.last_step_distance_threshold: # type: ignore
                self.update_target_data(self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"])
                last_action = [action[0], action[1], self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"]]
                
                self.total_last_step_distance += 1
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] WITHIN THE DIST, SETTING TARGET TO GOAL!" )
                success = self.client_set_action_drl(last_action, True)
            else:
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] REGULAR TARGET..." )
                success = self.client_set_action_drl(action)
  
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting mpc_action_complete for " + str(self.config_mobiman.action_time_horizon) + " sec...")
            time_start = time.time()
            while not self.mpc_action_complete:
                cmd = self.cmd_base + self.cmd_arm
                #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] cmd: " + str(cmd))
                self.robots[0].apply_action(cmd)
                self.simulator_step()
                #collision_links = self.run_simulation()
                #self.collision_links = collision_links
                #self.collision_step += int(len(collision_links) > 0)
            
            time_end = time.time()
            self.dt_action = time_end - time_start
            self.mpc_action_complete = False
            #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] Action completed in " + str(self.dt_action) + " sec!")

        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] DEBUG INF")
        #while 1:
        #    continue
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::take_action] END")

    '''
    DESCRIPTION: TODO...
    '''
    def is_done(self, observations):
        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] START")
        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] total_step_num: " + str(self.total_step_num))

        if self.step_num >= self.config_mobiman.max_episode_steps: # type: ignore
            self.termination_reason = 'max_step'
            self.total_max_step += 1
            self.episode_done = True
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Too late...")

        if self.episode_done and (not self.reached_goal):
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
        elif self.episode_done and self.reached_goal:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
        else:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")

        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] termination_reason: " + self.termination_reason) # type: ignore

        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] END")
        return self.episode_done

    '''
    DESCRIPTION: TODO...
    '''
    def compute_reward(self, observations, done):
        #print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] START")

        # 0: MPC/MRT Failure
        # 1: Collision
        # 2: Rollover
        # 3: Goal reached
        # 4: Target reached
        # 5: Time-horizon reached

        if self.episode_done and (not self.reached_goal):

            if self.termination_reason == 'collision':
                self.step_reward = self.config_mobiman.reward_terminal_collision
            elif self.termination_reason == 'rollover':
                self.step_reward = self.config_mobiman.reward_terminal_roll
            elif self.termination_reason == 'max_step':
                self.step_reward = self.config_mobiman.reward_terminal_max_step
            else:
                ### NUA NOTE: CHECKING IF THE ELSE CASE IS REACHABLE!
                self.step_reward = self.config_mobiman.reward_terminal_collision
                #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] CMOOOOOOOOOOOOOOON")
                #print("--------------------------------")
                #print("--------------------------------")
                #print("--------------------------------")
                #print("--------------------------------")
                #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG INF")
                #while 1:
                #    continue

            #self.goal_status.data = False
            #self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            self.training_data.append([self.episode_reward])

        elif self.episode_done and self.reached_goal:

            #self.step_reward = self.config_mobiman.reward_terminal_success + self.config_mobiman.reward_terminal_mintime * (self.config_mobiman.max_episode_steps - self.step_num) / self.config_mobiman.max_episode_steps
            self.step_reward = self.config_mobiman.reward_terminal_goal
            #self.goal_status.data = True
            #self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            self.training_data.append([self.episode_reward])

        else:
            # Step Reward 1: target to goal (considers both "previous vs. current" and "current target to goal")
            current_target2goal = self.get_euclidean_distance_3D(self.target_data, self.goal_data)
            reward_step_target2goal = self.reward_step_target2goal_func(current_target2goal, self.prev_target2goal)
            weighted_reward_step_target2goal = self.config_mobiman.alpha_step_target2goal * reward_step_target2goal # type: ignore
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] prev_target2goal: " + str(self.prev_target2goal))
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] current_target2goal: " + str(current_target2goal))
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] diff_target2goal: " + str(self.prev_target2goal - current_target2goal))
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] weighted_reward_step_target2goal: " + str(weighted_reward_step_target2goal))
            self.prev_target2goal = current_target2goal

            # Step Reward 2: model mode
            reward_step_mode = 0
            if self.model_mode == 0:
                reward_step_mode = self.config_mobiman.reward_step_mode0
            elif self.model_mode == 1:
                reward_step_mode = self.config_mobiman.reward_step_mode1
            elif self.model_mode == 2:
                reward_step_mode = self.config_mobiman.reward_step_mode2
            #else:
            #    print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] DEBUG INF")
            #    while 1:
            #        continue
            weighted_reward_step_mode = self.config_mobiman.alpha_step_mode * reward_step_mode # type: ignore

            # Step Reward 3: mpc result
            reward_step_mpc = 0
            if self.mpc_action_result == 0:
                reward_step_mpc = self.config_mobiman.reward_step_mpc_exit
            elif self.mpc_action_result == 4:
                reward_step_mpc = self.config_mobiman.reward_step_target_reached # type: ignore
            elif self.mpc_action_result == 5:
                reward_step_mpc = self.reward_step_time_horizon_func(self.dt_action)
                #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] dt_action: " + str(self.dt_action))
                #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] reward_step_mpc: " + str(reward_step_mpc))
            weighted_reward_mpc = self.config_mobiman.alpha_step_mpc_result * reward_step_mpc # type: ignore

            # Total Step Reward
            self.step_reward = weighted_reward_step_target2goal + weighted_reward_step_mode + weighted_reward_mpc

            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] reward_step: " + str(reward_step))
        
        self.episode_reward += self.step_reward # type: ignore

        if self.episode_done and self.episode_num > 0:
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_num: " + str(self.episode_num))
            #self.total_mean_episode_reward = round((self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num, self.config_mobiman.mantissa_precision)
            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] AND THE NEW total_mean_episode_reward!!!")
            self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num

            self.episode_num = self.episode_num + 1

            #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG INF")
            #while 1:
            #    continue

        self.save_oar_data()
        self.data = pd.DataFrame(self.oars_data)
        self.data.to_csv(self.oar_data_file)
        del self.data
        gc.collect()

        '''
        print("**********************")
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] done: " + str(done))
        #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] robot_id: {}".format(self.robot_id))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] step_num: {}".format(self.step_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_step_num: {}".format(self.total_step_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_num: {}".format(self.episode_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] ablation_mode: {}".format(self.config_mobiman.ablation_mode))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] mpc_action_result: {}".format(self.mpc_action_result))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] termination_reason: {}".format(self.termination_reason))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_collisions: {}".format(self.total_collisions))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_rollover: {}".format(self.total_rollover))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_goal: {}".format(self.total_goal))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_last_step_distance: {}".format(self.total_last_step_distance))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_max_step: {}".format(self.total_max_step))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_mpc_exit: {}".format(self.total_mpc_exit))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_target: {}".format(self.total_target))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_time_horizon: {}".format(self.total_time_horizon))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] step_reward: " + str(self.step_reward))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_reward: {}".format(self.episode_reward))
        print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
        print("**********************")
        '''

        '''
        # Save Observation-Action-Reward data into a file
        self.save_oar_data()

        if self.episode_done and (len(self.episode_oar_data['obs']) > 1):

            #print("[igibson_env_jackalJaco::iGibsonEnv::save_oar_data] episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
            #print("[igibson_env_jackalJaco::iGibsonEnv::save_oar_data] episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))

            if self.goal_status.data:
                info_data = np.ones(len(self.episode_oar_data['acts']))
            else:
                info_data = np.zeros(len(self.episode_oar_data['acts']))

            self.oar_data.append(TrajectoryWithRew( obs=np.array(self.episode_oar_data['obs']), 
                                                    acts=np.array(self.episode_oar_data['acts']),
                                                    infos=np.array(info_data),
                                                    terminal=True,
                                                    rews=np.array(self.episode_oar_data['rews']),))
        '''

        if self.total_step_num == self.config_mobiman.training_timesteps:
            
            # Write Observation-Action-Reward data into a file
            #self.write_oar_data()

            ## Write training data
            write_data(self.config_mobiman.data_folder_path + "training_data.csv", self.training_data) # type: ignore

        self.step_num += 1
        self.total_step_num += 1

        #print("[igibson_env_jackalJaco::iGibsonEnv::compute_reward] END")
        #print("--------------------------------------------------")
        #print("")
        return self.step_reward

    '''
    DESCRIPTION: Load task setup.
    '''
    def load_task_setup(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] START")

        self.init_ros_env(ros_node_id=self.ros_node_id, init_flag=self.init_ros_node)

        self.initial_pos_z_offset = self.config_igibson.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config_igibson.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config_igibson.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config_igibson.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config_igibson.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config_igibson.get("object_randomization_freq", None)

        # task
        if "task" not in self.config_igibson:
            self.task = DummyTask(self)
        elif self.config_igibson["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config_igibson["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config_igibson["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config_igibson["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config_igibson["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config_igibson["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        elif self.config_igibson["task"] == "mobiman_pick":
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] task: mobiman_pick")
            ### NUA TODO: SPECIFY NEW TASK ENVIRONMENT!
            self.task = DummyTask(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config_igibson["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config_igibson["task"]))
            except ImportError:
                raise Exception("bddl is not available.")

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] DEBUG INF")
        #while 1:
        #    continue

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] END")
    
    '''
    DESCRIPTION: Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
    '''
    def build_obs_space(self, shape, low, high):
        
        print("[igibson_env_jackalJaco::iGibsonEnv::build_obs_space] DEBUG_INF")
        while 1:
            continue
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    '''
    DESCRIPTION: Load observation space.
    '''
    def load_observation_space2(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] START")

        self.initialize_selfcoldistance_config()
        self.initialize_extcoldistance_base_config()
        self.initialize_extcoldistance_arm_config()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] DEBUG INF")
        #while 1:
        #    continue

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config_mobiman.observation_space_type == "mobiman_FC":
            
            '''
            self.initialize_occgrid_config()
            # Occupancy (OccupancyGrid data)
            if self.config_mobiman.occgrid_normalize_flag:   
                obs_occgrid_min = np.full((1, self.config_mobiman.occgrid_data_size), 0.0).reshape(self.config_mobiman.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config_mobiman.occgrid_data_size), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            else:
                obs_occgrid_min = np.full((1, self.config_mobiman.occgrid_data_size), self.config_mobiman.occgrid_occ_min).reshape(self.config_mobiman.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config_mobiman.occgrid_data_size), self.config_mobiman.occgrid_occ_max).reshape(self.config_mobiman.fc_obs_shape)
            '''

            # Self collision distances
            obs_selfcoldistance_min = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_selfcoldistance_max = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            '''
            # External collision distances (base to nearest objects)
            obs_extcoldistance_base_min = np.full((1, self.config_mobiman.n_extcoldistance_base), self.config_mobiman.ext_collision_range_base_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistance_base_max = np.full((1, self.config_mobiman.n_extcoldistance_base), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # External collision distances (from spheres on robot arm to nearest objects)
            obs_extcoldistance_arm_min = np.full((1, self.config_mobiman.n_extcoldistance_arm), self.config_mobiman.ext_collision_range_arm_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistance_arm_max = np.full((1, self.config_mobiman.n_extcoldistance_arm), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            '''

            # Collision object positions (wrt. robot base)
            obs_collision_base_min_single = np.array([[self.config_mobiman.world_range_x_min,
                                                       self.config_mobiman.world_range_y_min,
                                                       self.config_mobiman.world_range_z_min]])
            obs_collision_base_min = np.repeat(obs_collision_base_min_single, self.config_mobiman.n_extcoldistance_base, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            
            obs_collision_base_max_single = np.array([[self.config_mobiman.world_range_x_max,
                                                       self.config_mobiman.world_range_y_max,
                                                       self.config_mobiman.world_range_z_max]])
            obs_collision_base_max = np.repeat(obs_collision_base_max_single, self.config_mobiman.n_extcoldistance_base, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Collision object positions (wrt. robot base)
            obs_collision_arm_min_single = np.array([[self.config_mobiman.world_range_x_min,
                                                      self.config_mobiman.world_range_y_min,
                                                      self.config_mobiman.world_range_z_min]])
            obs_collision_arm_min = np.repeat(obs_collision_arm_min_single, self.config_mobiman.n_extcoldistance_arm, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            
            obs_collision_arm_max_single = np.array([[self.config_mobiman.world_range_x_max,
                                                      self.config_mobiman.world_range_y_max,
                                                      self.config_mobiman.world_range_z_max]])
            obs_collision_arm_max = np.repeat(obs_collision_arm_max_single, self.config_mobiman.n_extcoldistance_arm, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            # base x,y,z
            # ee x,y,z,roll,pitch,yaw
            obs_goal_min = np.array([[self.config_mobiman.world_range_x_min, # type: ignore
                                      self.config_mobiman.world_range_y_min, # type: ignore
                                      self.config_mobiman.world_range_z_min, 
                                      self.config_mobiman.world_range_x_min, # type: ignore
                                      self.config_mobiman.world_range_y_min, # type: ignore   
                                      self.config_mobiman.world_range_z_min, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config_mobiman.world_range_x_max, 
                                      self.config_mobiman.world_range_y_max, 
                                      self.config_mobiman.world_range_z_max, 
                                      self.config_mobiman.world_range_x_max, 
                                      self.config_mobiman.world_range_y_max, 
                                      self.config_mobiman.world_range_z_max, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Arm joint states
            obs_armstate_min = np.full((1, self.config_mobiman.n_armstate), -math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_armstate_max = np.full((1, self.config_mobiman.n_armstate), math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_occgrid_min shape: " + str(obs_occgrid_min.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_selfcoldistance_min shape: " + str(obs_selfcoldistance_min.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_collision_base_min shape: " + str(obs_collision_base_min.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_collision_arm_min shape: " + str(obs_collision_arm_min.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_goal_min shape: " + str(obs_goal_min.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_armstate_min shape: " + str(obs_armstate_min.shape))

            '''
            self.obs_data = {   "occgrid": np.vstack([obs_occgrid_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "selfcoldistance": np.vstack([obs_selfcoldistance_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_base": np.vstack([obs_collision_base_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_arm": np.vstack([obs_collision_arm_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "armstate": np.vstack([obs_armstate_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore
            '''
            
            self.obs_data = {   "selfcoldistance": np.vstack([obs_selfcoldistance_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_base": np.vstack([obs_collision_base_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_arm": np.vstack([obs_collision_arm_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "armstate": np.vstack([obs_armstate_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data selfcoldistance shape: " + str(self.obs_data["selfcoldistance"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data extcoldistance_base shape: " + str(self.obs_data["extcoldistance_base"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data extcoldistance_arm shape: " + str(self.obs_data["extcoldistance_arm"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data armstate shape: " + str(self.obs_data["armstate"].shape))

            #obs_stacked_occgrid_min = np.hstack([obs_occgrid_min] * self.config_mobiman.n_obs_stack) # type: ignore
            #obs_stacked_occgrid_max = np.hstack([obs_occgrid_max] * self.config_mobiman.n_obs_stack) # type: ignore

            #obs_space_min = np.concatenate((obs_stacked_occgrid_min, obs_extcoldistancedist_min, obs_goal_min), axis=0)
            #obs_space_max = np.concatenate((obs_stacked_occgrid_max, obs_extcoldistancedist_max, obs_goal_max), axis=0)

            obs_space_min = np.concatenate((obs_selfcoldistance_min, obs_collision_base_min, obs_collision_arm_min, obs_goal_min, obs_armstate_min), axis=0)
            obs_space_max = np.concatenate((obs_selfcoldistance_max, obs_collision_base_max, obs_collision_arm_max, obs_goal_max, obs_armstate_max), axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_stacked_laser_low shape: " + str(obs_stacked_laser_low.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_space_min shape: " + str(obs_space_min.shape))

            self.obs = obs_space_min
            self.observation_space = gym.spaces.Box(obs_space_min, obs_space_max)

        elif self.config_mobiman.observation_space_type == "mobiman_2DCNN_FC":

            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] NEEDS REVIEW: DEBUG INF")
            while 1:
                continue

            self.initialize_occgrid_config()

            # Occupancy (OccupancyGrid image)
            obs_occgrid_image_min = np.full((1, self.config_mobiman.occgrid_width), 0.0)
            obs_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config_mobiman.occgrid_height)
            obs_occgrid_image_min = np.expand_dims(obs_occgrid_image_min, axis=0)

            obs_occgrid_image_max = np.full((1, self.config_mobiman.occgrid_width), 1.0)
            obs_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config_mobiman.occgrid_height)
            obs_occgrid_image_max = np.expand_dims(obs_occgrid_image_max, axis=0)

            # Nearest collision distances (from spheres on robot body)
            obs_extcoldistancedist_min = np.full((1, self.config_mobiman.n_extcoldistance), self.config_mobiman.ext_collision_range_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistancedist_max = np.full((1, self.config_mobiman.n_extcoldistance), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            obs_goal_min = np.array([[self.config_mobiman.goal_range_min_x, # type: ignore
                                      self.config_mobiman.goal_range_min_y, # type: ignore
                                      self.config_mobiman.goal_range_min_z, 
                                      self.config_mobiman.goal_range_min_x, # type: ignore
                                      self.config_mobiman.goal_range_min_y, # type: ignore
                                      self.config_mobiman.goal_range_min_z, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config_mobiman.goal_range_max_x, 
                                      self.config_mobiman.goal_range_max_y, 
                                      self.config_mobiman.goal_range_max_z, 
                                      self.config_mobiman.goal_range_max_x, 
                                      self.config_mobiman.goal_range_max_y, 
                                      self.config_mobiman.goal_range_max_z, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_occgrid_image_min shape: " + str(obs_occgrid_image_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_extcoldistancedist_min shape: " + str(obs_extcoldistancedist_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_goal_min shape: " + str(obs_goal_min.shape))

            self.obs_data = {   "occgrid_image": np.vstack([obs_occgrid_image_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistancedist": np.vstack([obs_extcoldistancedist_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore

            obs_space_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config_mobiman.n_obs_stack) # type: ignore
            obs_space_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config_mobiman.n_obs_stack) # type: ignore

            obs_space_extcoldistancedist_goal_min = np.concatenate((obs_extcoldistancedist_min, obs_goal_min), axis=0)
            obs_space_extcoldistancedist_goal_max = np.concatenate((obs_extcoldistancedist_max, obs_goal_max), axis=0)

            self.obs = {"occgrid_image": obs_space_occgrid_image_min, 
                        "extcoldistancedist_goal": obs_space_extcoldistancedist_goal_min}

            self.observation_space = gym.spaces.Dict({  "occgrid_image": gym.spaces.Box(obs_space_occgrid_image_min, obs_space_occgrid_image_max), 
                                                        "extcoldistancedist_goal": gym.spaces.Box(obs_space_extcoldistancedist_goal_min, obs_space_extcoldistancedist_goal_max)})

        self.config_mobiman.set_observation_shape(self.observation_space.shape)

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] observation_space shape: " + str(self.observation_space.shape))
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] observation_space: " + str(self.observation_space))
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] END")

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] DEBUG INF")
        #while 1:
        #   continue

    '''
    DESCRIPTION: Load observation space.
    '''
    def load_observation_space(self):
        """
        Load observation space.
        """
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] DEPRECATED DEBUG_INF")
        while 1:
            continue

        #self.output = self.config_igibson["output"]
        #self.image_width = self.config_igibson.get("image_width", 128)
        #self.image_height = self.config_igibson.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("depth")
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "highlight" in self.output:
            observation_space["highlight"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("highlight")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config_igibson.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config_igibson.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "scan_rear" in self.output:
            self.n_horizontal_rays = self.config_igibson.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config_igibson.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan_rear"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan_rear")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config_igibson.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")
        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)
        if "proprioception" in self.output:
            observation_space["proprioception"] = self.build_obs_space(
                shape=(self.robots[0].proprioception_dim,), low=-np.inf, high=np.inf
            )

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        if "scan_rear" in scan_modalities:
            sensors["scan_occ_rear"] = ScanSensor(self, scan_modalities, rear=True)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] END")

    '''
    DESCRIPTION: Load action space.
    '''
    def load_action_space(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] START")

        #self.action_space = self.robots[0].action_space

        if self.config_mobiman.action_type == 0:
            self.action_space = gym.spaces.Discrete(self.config_mobiman.n_action)
        else:
            action_space_model_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_constraint_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_min = np.array([self.config_mobiman.goal_range_min_x, self.config_mobiman.goal_range_min_y, self.config_mobiman.goal_range_min_z]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            action_space_target_ori_min = np.full((1, 3), -math.pi).reshape(self.config_mobiman.fc_obs_shape)
            obs_space_min = np.concatenate((action_space_model_min, action_space_constraint_min, action_space_target_pos_min, action_space_target_ori_min), axis=0)

            action_space_model_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_constraint_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_max = np.array([self.config_mobiman.goal_range_max_x, self.config_mobiman.goal_range_max_y, self.config_mobiman.goal_range_max_z]).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_ori_max = np.full((1, 3), math.pi).reshape(self.config_mobiman.fc_obs_shape)
            obs_space_max = np.concatenate((action_space_model_max, action_space_constraint_max, action_space_target_pos_max, action_space_target_ori_max), axis=0)
            
            self.action_space = gym.spaces.Box(obs_space_min, obs_space_max)

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_type: " + str(self.config_mobiman.action_type))
        if self.config_mobiman.action_type == 0:
            self.config_mobiman.set_action_shape("Discrete, " + str(self.action_space.n)) # type: ignore
        else:
            self.config_mobiman.set_action_shape(self.action_space.shape)
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space shape: " + str(self.action_space.shape))
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space: " + str(self.action_space))

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space: " + str(self.action_space))
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] DEBUG INF")
        #while 1:
        #    continue
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] END")

    '''
    DESCRIPTION: Load miscellaneous variables for book keeping.
    '''
    def load_miscellaneous_variables(self):
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    '''
    DESCRIPTION: Load environment.
    '''
    def load(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START")

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START super")
        super(iGibsonEnv, self).load()
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_task_setup")
        self.load_task_setup()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_observation_space")
        self.load_observation_space2()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_action_space")
        self.load_action_space()
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_miscellaneous_variables")
        self.load_miscellaneous_variables()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] DEBUG_INF")   
        #while 1:
        #    continue

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] END")

    '''
    DESCRIPTION: Get the current observation.

        :return: observation as a dictionary
    '''
    def get_state(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] DEBUG_INF")   
        while 1:
            continue

        state = OrderedDict()

        '''
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)

        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "scan_occ_rear" in self.sensors:
            scan_obs = self.sensors["scan_occ_rear"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)
        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())
        '''

        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] END")

        return state

    '''
    DESCRIPTION: Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
    '''
    def run_simulation(self):
        self.simulator_step()
        collision_links = [
            collision for bid in self.robots[0].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored.

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        # TODO: Improve this to accept multi-body robots.
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        :param info: the info dictionary to populate
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] START")

        #print("[igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")   
        #while 1:
        #    continue

        
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")   
        #while 1:
        #    continue

        info = {}

        # Take action
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE take_action")
        self.take_action(action)

        # Update data
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE update_data")
        self.update_robot_data()
        self.update_arm_data()
        self.update_goal_data()
        self.update_goal_data_wrt_robot()
        self.update_goal_data_wrt_ee()
        #self.update_target_data()

        # Update observation (state)
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE update_observation")
        self.update_observation()
        state = self.obs

        # Check if episode is done
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE is_done")
        done = self.is_done(state)

        # Compute reward
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE compute_reward")
        reward = self.compute_reward(state, done)

        #print("[igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")
        #while 1:
        #    continue

        '''
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE INIT action: " + str(action))
        action = self.cmd_init_base + self.cmd_init_arm
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] AFTER INIT action: " + str(action))
        #print("")
        action[0] = 0.1

        self.current_step += 1
        #if action is not None:
        self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        #print("[igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")
        #while 1:
        #    continue

        #state = self.get_state()
        state = np.full((1, 56), 0.0).reshape(-1) # type: ignore
        info = {}
        reward =  0.0
        done = False
        success = False
        #reward, info = self.task.get_reward(self, collision_links, action, info)
        #done, info = self.task.get_termination(self, collision_links, action, info)
        #self.task.step(self)
        self.populate_info(info)
        '''

        if done and self.automatic_reset:

            print("[igibson_env_jackalJaco::iGibsonEnv::step] EPISODE DONE for " + str(self.ns))
            print("[igibson_env_jackalJaco::iGibsonEnv::step] episode_num: " + str(self.episode_num))

            info["last_observation"] = state
            state = self.reset()

        return state, reward, done, info

    def check_collision(self, body_id, ignore_ids=[]):
        """
        Check whether the given body_id has collision after one simulator step

        :param body_id: pybullet body id
        :param ignore_ids: pybullet body ids to ignore collisions with
        :return: whether the given body_id has collision
        """
        self.simulator_step()
        collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]

        if log.isEnabledFor(logging.INFO):  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                log.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) > 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        ignore_ids = obj.get_body_ids() if ignore_self_collision else []
        has_collision = any(self.check_collision(body_id, ignore_ids) for body_id in obj.get_body_ids())
        return not has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if not land_success:
            log.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

        #goal_frame_name = self.ns + self.config_mobiman.goal_frame_name

        #print("[igibson_env_jackalJaco::iGibsonEnv::reset_variables] Waiting for transformation of goal wrt world...")
        #print("[igibson_env_jackalJaco::iGibsonEnv::reset_variables] world_frame_name: " + str(self.config_mobiman.world_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::reset_variables] goal_frame_name: " + str(goal_frame_name))
        #self.listener.waitForTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=goal_frame_name, time=rospy.Time(0), timeout=rospy.Duration(100))
        #rospy.wait_for_message(self.config_mobiman)
        #print("[igibson_env_jackalJaco::iGibsonEnv::reset_variables] Received transform between base and goal!")

        #print("[igibson_env_jackalJaco::iGibsonEnv::reset_variables] DEBUG_INF")
        #while 1:
        #    continue

        if self.init_flag:
            self.update_robot_data()
            self.update_arm_data()
            self.update_goal_data()
            self.update_goal_data_wrt_robot()
            self.update_goal_data_wrt_ee()
            #self.update_target_data()
            self.previous_base_distance2goal = self.get_base_distance2goal_2D()
            self.prev_target2goal = self.get_base_distance2goal_2D()

    '''
    DESCRIPTION: TODO...Gets the initial location of the robot to reset
    '''
    def initialize_robot_pose(self):
        robot0_init_yaw = 0.0
        if self.config_mobiman.world_name == "conveyor":

            '''
            # Set world range
            self.world_range_x_min = -5.0
            self.world_range_x_max = 5.0

            self.world_range_y_min = -5.0
            self.world_range_y_max = 5.0

            self.world_range_z_min = 0.0
            self.world_range_z_max = 2.0
            '''

            p1 = {"x":self.config_mobiman.world_range_x_min, "y":self.config_mobiman.world_range_y_min, "z":self.config_mobiman.world_range_z_min}
            p2 = {"x":self.config_mobiman.world_range_x_max, "y":self.config_mobiman.world_range_y_max, "z":self.config_mobiman.world_range_z_max}

            #self.world_diameter = self.get_euclidean_distance_3D(p1, p2)

            # Set init robot pose
            init_robot_pose_areas_x = []
            init_robot_pose_areas_x.extend(([-2.0,2.0], [-2.0,2.0], [-2.0,-2.0], [-2.0,2.0]))

            init_robot_pose_areas_y = []
            init_robot_pose_areas_y.extend(([-1.5,2.0], [-1.5,2.0], [-1.5,2.0], [-1.5,2.0]))

            area_idx = random.randint(0, len(init_robot_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.init_robot_pose["x"] = random.uniform(init_robot_pose_areas_x[area_idx][0], init_robot_pose_areas_x[area_idx][1])
            self.init_robot_pose["y"] = random.uniform(init_robot_pose_areas_y[area_idx][0], init_robot_pose_areas_y[area_idx][1])
            self.init_robot_pose["z"] = 0.15
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] x: " + str(self.init_robot_pose["x"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] y: " + str(self.init_robot_pose["y"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] z: " + str(self.init_robot_pose["z"]))

        robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
        self.init_robot_pose["qx"] = robot0_init_quat.x
        self.init_robot_pose["qy"] = robot0_init_quat.y
        self.init_robot_pose["qz"] = robot0_init_quat.z
        self.init_robot_pose["qw"] = robot0_init_quat.w

        self.init_joint_states = {}
        init_arm_joint_pos = [0.0, 2.9, 1.3, 4.2, 1.4, 0.0]
        init_arm_joint_velo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, jn in enumerate(self.config_mobiman.arm_joint_names):
            self.init_joint_states[jn] = (init_arm_joint_pos[i], init_arm_joint_velo[i])

        #print("[igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] init_joint_states:")
        #print(self.init_joint_states)

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] Updated init_robot_pose x: " + str(self.init_robot_pose["x"]) + ", y: " + str(self.init_robot_pose["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] Updated init_robot_pose x: " + str(self.init_robot_pose["x"]) + ", y: " + str(self.init_robot_pose["y"]))

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

        self.initialize_robot_pose()

    def reset(self):
        """
        Reset episode.
        """
        print("[igibson_env_jackalJaco::iGibsonEnv::reset] START")
        print("==================================================")
        print("==================================================")
        print("==================================================")
        print("==================================================")
        print("==================================================")
        print("==================================================")
        print("==================================================")
        
        self.randomize_domain()

        #print("[igibson_env_jackalJaco::iGibsonEnv::reset] DEBUG_INF")
        #while 1:
        #    continue

        # Move robot away from the scene.
        #self.robots[0].set_position([100.0, 100.0, 100.0])
        ### NUA TODO: RANDOMIZE!
        #self.robots[0].set_position([0.0, 0.0, 0.0])
        init_robot_pos = [self.init_robot_pose["x"], self.init_robot_pose["y"], self.init_robot_pose["z"]]
        init_robot_quat = [self.init_robot_pose["qx"], self.init_robot_pose["qy"], self.init_robot_pose["qz"], self.init_robot_pose["qw"]]
        self.robots[0].set_position_orientation(init_robot_pos, init_robot_quat)
        self.robots[0].set_joint_states(self.init_joint_states)
        
        self.cmd = self.cmd_init_base + self.cmd_init_arm

        #print("[igibson_env_jackalJaco::iGibsonEnv::reset] DEBUG_INF")
        #while 1:
        #    continue

        ### NUA TODO: UTILIZE THIS!
        self.task.reset(self)
        
        self.simulator.sync(force_sync=True)

        self.reset_variables()

        self.update_observation()
        state = self.obs

        #print("[igibson_env_jackalJaco::iGibsonEnv::reset] DEBUG_INF")
        #while 1:
        #    continue

        print("[igibson_env_jackalJaco::iGibsonEnv::reset] END")

        return state
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_euclidean_distance_2D(self, p1, p2={"x":0.0, "y":0.0}):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

    '''
    DESCRIPTION: TODO...
    '''
    def get_euclidean_distance_3D(self, p1, p2={"x":0.0, "y":0.0, "z":0.0}):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2 + (p1["z"] - p2["z"])**2)
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_quaternion_distance(self, q1, q2={"qx":0.0, "qy":0.0, "qz":0.0, "qw":1.0}):
        m_q1 = np.array([q1["qx"], q1["qy"], q1["qz"]])
        m_q2 = np.array([q2["qx"], q2["qy"], q2["qz"]])
        qdist_vec = q1["qw"] * m_q2 + q2["qw"] * m_q1 + np.cross(m_q1, m_q2) # type: ignore
        qdist = np.linalg.norm(qdist_vec)
        #print("[igibson_env_jackalJaco::iGibsonEnv::get_quaternion_distance] qdist: " + str(qdist))
        return qdist

    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.init_robot_pose)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.init_robot_pose)
        return distance2goal

    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.robot_data)

        pp1 = Point()
        pp1.x = self.goal_data["x"]
        pp1.y = self.goal_data["y"]
        pp1.z = self.goal_data["z"]

        pp2 = Point()
        pp2.x = self.robot_data["x"]
        pp2.y = self.robot_data["y"]
        pp2.z = self.robot_data["z"]
 
        debug_point_data = [pp1, pp2]
        for i in range(1, 100):
            self.publish_debug_visu(debug_point_data)

        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.robot_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_arm_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.arm_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_arm_quatdistance2goal(self):
        distance2goal = self.get_quaternion_distance(self.goal_data, self.arm_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_base_distance2target_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.target_data, self.robot_data)
        return distance2goal

    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_arm_distance2target_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.target_data, self.arm_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_base_yawdistance2target(self):
        #distance2goal = abs(self.target_data["yaw"] - self.robot_data["yaw"])

        quat1 = [self.target_data["qx"], self.target_data["qy"], self.target_data["qz"], self.target_data["qw"]]
        quat2 = [self.robot_data["qx"], self.robot_data["qy"], self.robot_data["qz"], self.robot_data["qw"]]

        euler1 = tftrans.euler_from_quaternion(quat1) # type: ignore
        euler2 = tftrans.euler_from_quaternion(quat2) # type: ignore
        
        yaw1 = euler1[2]
        yaw2 = euler2[2]

        yaw_diff = yaw2 - yaw1

        # Normalize yaw difference to be within range of -pi to pi
        while yaw_diff > math.pi:
            yaw_diff -= 2*math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2*math.pi

        return abs(yaw_diff / math.pi)

    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_arm_quatdistance2target(self):
        distance2goal = self.get_quaternion_distance(self.target_data, self.arm_data)

        '''
        if self.model_mode != 0 and (distance2goal > 1 or distance2goal < 0):
            print("[igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] model_mode: " + str(self.model_mode))
            print("[igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] distance2goal: " + str(distance2goal))

            print("[igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] DEBUG INF")
            while 1:
                continue
        '''
        return distance2goal
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_occgrid(self, image_flag=False):
        if image_flag:
            obs_occgrid = self.get_occgrid_image()
        else:
            occgrid_msg = self.occgrid_msg
            obs_occgrid = np.asarray(occgrid_msg.data)
            if self.config_mobiman.occgrid_normalize_flag:
                max_scale = 1 / self.config_mobiman.occgrid_occ_max # type: ignore
                obs_occgrid = max_scale * obs_occgrid
            obs_occgrid = obs_occgrid.reshape(self.config_mobiman.fc_obs_shape)
        return obs_occgrid

    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_selfcoldistance(self):
        selfcoldistance_msg = self.selfcoldistance_msg

        obs_selfcoldistance = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        for i, dist in enumerate(selfcoldistance_msg.distance):
            #csm = selfcoldistance_msg.markers[i*self.config_mobiman.selfcoldistance_n_coeff] # type: ignore
            #p1 = {"x":csm.points[0].x, "y":csm.points[0].y, "z":csm.points[0].z}
            #p2 = {"x":csm.points[1].x, "y":csm.points[1].y, "z":csm.points[1].z} 
            #dist = self.get_euclidean_distance_3D(p1, p2)
            obs_selfcoldistance[i] = dist
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] dist " + str(i) + ": " + str(dist))
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] obs_selfcoldistance shape: " + str(obs_selfcoldistance.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] DEBUG INF")
        #while 1:
        #    continue
        
        return obs_selfcoldistance
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_extcoldistance_base(self):
        extcoldistance_base_msg = self.extcoldistance_base_msg

        #debug_point_data = []
        #obs_extcoldistance_base = np.full((1, self.config_mobiman.n_extcoldistance_base), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_extcoldistance_base = np.array([[extcoldistance_base_msg.p1[0].x, 
                                             extcoldistance_base_msg.p1[0].y, 
                                             extcoldistance_base_msg.p1[0].z]]).reshape(self.config_mobiman.fc_obs_shape)
        for i, p1 in enumerate(extcoldistance_base_msg.p1):
            #p1 = {"x":csm.points[0].x, "y":csm.points[0].y, "z":csm.points[0].z}
            #p2 = {"x":csm.points[1].x, "y":csm.points[1].y, "z":csm.points[1].z}
            #dist = self.get_euclidean_distance_3D(p1, p2)
            #obs_extcoldistance_base[i] = dist
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_base] p" + str(i) + ".x: " + str(p1.x))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_base] p" + str(i) + ".y: " + str(p1.y))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_base] p" + str(i) + ".z: " + str(p1.z))

            '''
            p = Point()
            p.x = p1.x
            p.y = p1.y
            p.z = p1.z
            debug_point_data.append(p)
            '''

            if i > 0:
                obs_extcoldistance_base_tmp = np.array([[p1.x, p1.y, p1.z]]).reshape(self.config_mobiman.fc_obs_shape)
                obs_extcoldistance_base = np.concatenate((obs_extcoldistance_base, obs_extcoldistance_base_tmp), axis=0)
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_base] obs_extcoldistance_base shape: " + str(obs_extcoldistance_base.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_base] DEBUG INF")
        #while 1:
        #    continue

        #self.publish_debug_visu(debug_point_data, extcoldistance_base_msg.frame_name)
        
        return obs_extcoldistance_base
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_extcoldistance_arm(self):
        extcoldistance_arm_msg = self.extcoldistance_arm_msg

        #debug_point_data = []
        #obs_extcoldistance_arm = np.full((1, self.config_mobiman.n_extcoldistance_arm), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_extcoldistance_arm = np.array([[extcoldistance_arm_msg.p1[0].x, 
                                            extcoldistance_arm_msg.p1[0].y, 
                                            extcoldistance_arm_msg.p1[0].z]]).reshape(self.config_mobiman.fc_obs_shape)
        for i, p1 in enumerate(extcoldistance_arm_msg.p1):
            #p1 = {"x":csm.points[0].x, "y":csm.points[0].y, "z":csm.points[0].z}
            #p2 = {"x":csm.points[1].x, "y":csm.points[1].y, "z":csm.points[1].z}
            #dist = self.get_euclidean_distance_3D(p1, p2)
            #obs_extcoldistance_arm[i] = dist
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_arm] p" + str(i) + ".x: " + str(p1.x))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_arm] p" + str(i) + ".y: " + str(p1.y))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_arm] p" + str(i) + ".z: " + str(p1.z))

            '''
            p = Point()
            p.x = p1.x
            p.y = p1.y
            p.z = p1.z
            debug_point_data.append(p)
            '''

            if i > 0:
                obs_extcoldistance_arm_tmp = np.array([[p1.x, p1.y, p1.z]]).reshape(self.config_mobiman.fc_obs_shape)
                obs_extcoldistance_arm = np.concatenate((obs_extcoldistance_arm, obs_extcoldistance_arm_tmp), axis=0)

        #self.publish_debug_visu(debug_point_data, extcoldistance_arm_msg.frame_name)

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_arm] obs_extcoldistance_arm shape: " + str(obs_extcoldistance_arm.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_extcoldistance_arm] DEBUG INF")
        #while 1:
        #    continue
        
        return obs_extcoldistance_arm
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_armstate(self):
        #extcoldistance_arm_msg = self.extcoldistance_arm_msg

        obs_armstate = np.full((1, self.config_mobiman.n_armstate), 0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        for i, jp in enumerate(self.arm_data["joint_pos"]):
            obs_armstate[i] = jp

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] obs_armstate shape: " + str(obs_armstate.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] DEBUG INF")
        #while 1:
        #    continue
        
        return obs_armstate
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_pointsonrobot(self):
        pointsonrobot_msg = self.pointsonrobot_msg

        obs_pointsonrobot = []
        for i, pm in enumerate(pointsonrobot_msg.markers):
            if i != 0:
                p = Point()
                p.x = pm.pose.position.x
                p.y = pm.pose.position.y
                p.z = pm.pose.position.z
                obs_pointsonrobot.append(p)
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_pointsonrobot] obs_extcoldistancedist shape: " + str(obs_extcoldistancedist.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_pointsonrobot] DEBUG INF")
        #while 1:
        #    continue
        
        return obs_pointsonrobot

    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_goal(self):
        quat_goal_wrt_ee = Quaternion(self.goal_data["qw_wrt_ee"], self.goal_data["qx_wrt_ee"], self.goal_data["qy_wrt_ee"], self.goal_data["qz_wrt_ee"])
        euler_goal_wrt_ee = quat_goal_wrt_ee.to_euler(degrees=False)

        obs_goal = np.array([[self.goal_data["x_wrt_robot"], 
                              self.goal_data["y_wrt_robot"], 
                              self.goal_data["z_wrt_robot"],
                              self.goal_data["x_wrt_ee"], 
                              self.goal_data["y_wrt_ee"], 
                              self.goal_data["z_wrt_ee"],
                              euler_goal_wrt_ee[0],
                              euler_goal_wrt_ee[1],
                              euler_goal_wrt_ee[2]]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        return obs_goal

    '''
    DESCRIPTION: TODO...
    '''
    def linear_function(self, x_min, x_max, y_min, y_max, query_x, slope_sign=1):
        if x_min <= query_x <= x_max:
            slope = slope_sign * (y_max - y_min) / (x_max - x_min)
            y_intercept = y_max - slope * x_min
            return slope * query_x + y_intercept
        else:
            if query_x < x_min:
                if slope_sign < 0:
                    return y_max
                else:
                    return y_min
            else:
                if slope_sign < 0:
                    return y_min
                else:
                    return y_max
                
    '''
    DESCRIPTION: TODO...
    '''     
    def sigmoid_function(self, x, gamma):
        return (1 / (1 + np.exp(-gamma * x))) # type: ignore

    '''
    DESCRIPTION: TODO...
    '''
    def gaussian_function(self, x, sigma):
        """ Return the scaled Gaussian with standard deviation sigma. """
        gaussian = np.exp(- (x / sigma)**2)
        scaled_result = 2 * gaussian - 1
        return scaled_result

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_target2goal_diff_func(self, curr_target2goal, prev_target2goal):
        diff_target2goal = prev_target2goal - curr_target2goal
        reward_step_target2goal = self.config_mobiman.reward_step_target2goal * self.gaussian_function(diff_target2goal-self.config_mobiman.reward_step_target2goal_mu_regular, self.config_mobiman.reward_step_target2goal_sigma_regular)
        return reward_step_target2goal

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_target2goal_curr_func(self, curr_target2goal):
        reward_step_target2goal = self.config_mobiman.reward_step_target2goal * self.gaussian_function(curr_target2goal-self.config_mobiman.reward_step_target2goal_mu_last_step, self.config_mobiman.reward_step_target2goal_sigma_last_step)
        return reward_step_target2goal
    
    def reward_step_target2goal_func(self, curr_target2goal, prev_target2goal):
        distance2goal = self.get_base_distance2goal_2D()
        if distance2goal < self.config_mobiman.last_step_distance_threshold: # type: ignore
            print("[igibson_env_jackalJaco::iGibsonEnv::reward_step_target2goal_func] WITHIN LAST STEP DISTANCE!")
            return self.reward_step_target2goal_curr_func(curr_target2goal)
        else:
            return self.reward_step_target2goal_diff_func(curr_target2goal, prev_target2goal)

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_time_horizon_func(self, dt_action):
        reward_step_mpc_time_horizon = 0
        if dt_action <= self.config_mobiman.action_time_horizon:
            reward_step_mpc_time_horizon = self.linear_function(0.0, self.config_mobiman.action_time_horizon, 
                                                                0.0, self.config_mobiman.reward_step_time_horizon_max, 
                                                                dt_action, slope_sign=-1) # type: ignore
        elif dt_action <= 2*self.config_mobiman.action_time_horizon: # type: ignore
            reward_step_mpc_time_horizon = self.linear_function(self.config_mobiman.action_time_horizon, 2*self.config_mobiman.action_time_horizon,  # type: ignore
                                                                   self.config_mobiman.reward_step_time_horizon_min, 0.0, 
                                                                   dt_action, slope_sign=-1) # type: ignore
        else:
            if  dt_action > 2*self.config_mobiman.action_time_horizon: # type: ignore
                reward_step_mpc_time_horizon = self.config_mobiman.reward_step_time_horizon_min
            else:
                reward_step_mpc_time_horizon = self.config_mobiman.reward_step_time_horizon_max
        return reward_step_mpc_time_horizon

    '''
    DESCRIPTION: TODO...
    '''
    def check_collision(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] START")
        
        print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] DEBUG_INF")
        while 1:
            continue

        selfcoldistance = self.get_obs_selfcoldistance() 
        extcoldistance_base = self.get_obs_extcoldistance_base()
        extcoldistance_arm = self.get_obs_extcoldistance_arm()
        pointsonrobot = self.get_pointsonrobot()
        
        for dist in selfcoldistance:
            #print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] selfcoldistance dist: " + str(dist))
            if dist < self.config_mobiman.self_collision_range_min:
                print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] SELF COLLISION")
                self.episode_done = True
                self.termination_reason = 1
                return True
            
        for dist in extcoldistance_base:
            #print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] extcoldistance_base dist: " + str(dist))
            if dist < self.config_mobiman.ext_collision_range_base_min:
                print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] EXT BASE COLLISION")
                self.episode_done = True
                self.termination_reason = 1
                return True
            
        for dist in extcoldistance_arm:
            #print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] extcoldistance_arm dist: " + str(dist))
            if dist < self.config_mobiman.ext_collision_range_arm_min:
                print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] EXT ARM COLLISION")
                self.episode_done = True
                self.termination_reason = 1
                return True

        for por in pointsonrobot:
            if por.z < self.config_mobiman.ext_collision_range_base_min:
                print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] GROUND COLLISION ")
                self.episode_done = True
                self.termination_reason = 1
                return True

        #print("[igibson_env_jackalJaco::iGibsonEnv::check_collision] END")

        return False
    
    '''
    DESCRIPTION: TODO...
    value.
    '''
    def check_rollover(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] START")
        
        print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] DEBUG_INF")
        while 1:
            continue

        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] pitch: " + str(self.robot_data["pitch"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] rollover_pitch_threshold: " + str(self.config_mobiman.rollover_pitch_threshold))
        # Check pitch
        if self.robot_data["pitch"] > self.config_mobiman.rollover_pitch_threshold:
            print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] PITCH ROLLOVER!!!")
            self.episode_done = True
            self.termination_reason = 2
            return True
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] roll: " + str(self.robot_data["roll"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] rollover_roll_threshold: " + str(self.config_mobiman.rollover_roll_threshold))
        # Check roll
        if self.robot_data["roll"] > self.config_mobiman.rollover_roll_threshold:
            print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] ROLL ROLLOVER!!!")
            self.episode_done = True
            self.termination_reason = 2
            return True
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::check_rollover] END")

        return False
    
    '''
    DESCRIPTION: TODO...
    '''
    def publish_debug_visu(self, debug_point_data, frame_name=""):

        if frame_name == "":
            frame_name = self.config_mobiman.world_frame_name
        
        debug_visu = MarkerArray()
        for i, dp in enumerate(debug_point_data):
            marker = Marker()
            marker.header.frame_id = frame_name
            marker.ns = str(i)
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = dp.x
            marker.pose.position.y = dp.y
            marker.pose.position.z = dp.z
            marker.header.stamp = rospy.Time.now()

            debug_visu.markers.append(marker) # type: ignore
            
        self.debug_visu_pub.publish(debug_visu) # type: ignore

    '''
    DESCRIPTION: TODO...
    '''
    def save_oar_data(self):
        if  self.config_mobiman.observation_space_type == "laser_FC" or \
            self.config_mobiman.observation_space_type == "Tentabot_FC" or \
            self.config_mobiman.observation_space_type == "mobiman_FC":
        
                #print("----------------------------------")
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] self.obs shape: " + str(self.obs.shape))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] self.previous_action shape: " + str(self.previous_action.shape))
                #print("")

                obs_data = self.obs.reshape((-1)) # type: ignore
                
                
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] obs_data shape: " + str(obs_data.shape))
                #print("----------------------------------")

                # Save Observation-Action-Reward Data
                self.episode_oar_data['obs'].append(obs_data) # type: ignore
                self.oars_data['Index'].append(self.idx)
                self.oars_data['Observation'].append(obs_data.tolist())
                self.oars_data['Action'].append(self.step_action)
                self.oars_data['Reward'].append(self.step_reward)
                if not self.episode_done:
                    self.episode_oar_data['acts'].append(self.action_space) # type: ignore
                    #self.episode_oar_data['infos'].append()
                    #self.episode_oar_data['terminal'].append(self.episode_done)
                    self.episode_oar_data['rews'].append(self.step_reward) # type: ignore
                    ############ CSV #################
                else:
                    # self.episode_oar_data['obs'].append(obs_data) # type: ignore
                    self.oars_data['Index'].append(None)
                    self.oars_data['Observation'].append([])
                    self.oars_data['Action'].append([])
                    self.oars_data['Reward'].append([])
                    self.idx = 0
                self.idx += 1

                '''
                print("----------------------------------")
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs type: " + str(type(self.episode_oar_data['obs'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data: " + str(self.episode_oar_data))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs: " + str(self.episode_oar_data.obs))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs shape: " + str(self.episode_oar_data.obs.shape))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] oar_data: " + str(self.oar_data))
                print("----------------------------------")
                '''

    '''
    DESCRIPTION: TODO...Save a sequence of Trajectories.

        Args:
            path: Trajectories are saved to this path.
            trajectories: The trajectories to save.
    '''
    def write_oar_data(self) -> None:
        path = self.config_mobiman.data_folder_path + "oar_data.pkl"
        trajectories = self.oar_data
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump(trajectories, f)

        # Ensure atomic write
        os.replace(tmp_path, path)

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::write_oar_data] Written Observation-Action-Reward data!")