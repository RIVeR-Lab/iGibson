#!/usr/bin/python3

'''
LAST UPDATE: 2024.03.14

AUTHOR: Neset Unver Akmandor (NUA)
        Sarvesh Prajapati (SP)

E-MAIL: akmandor.n@northeastern.edu
        prajapati.s@northeastern.edu

DESCRIPTION: TODO...

REFERENCES: iGibson

NUA TODO:
- 
'''

import argparse
from ast import Continue
import logging
from logging import config
import os
import time
import math
import random
from matplotlib.pyplot import flag
from networkx import configuration_model
from rsa import sign
from sympy import false, true
import yaml

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
from typing import Optional

#import gym
import gymnasium as gym

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
from std_msgs.msg import Header, UInt8
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import MarkerArray, Marker

from ocs2_msgs.msg import mpc_data, collision_info, MobimanGoalObservation, MobimanOccupancyObservation # type: ignore 
from ocs2_msgs.srv import calculateMPCTrajectory, setDiscreteActionDRL, setContinuousActionDRL, setBool, setBoolResponse, setMPCActionResult, setMPCActionResultResponse # type: ignore

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

    """
    ### DESCRIPTION: NUA TODO: UPDATE!
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
    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        drl_mode="training",
        action_timestep=1/5.0,
        physics_timestep=1/60.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
        init_ros_node=False,
        ros_node_id=0,
        data_folder_path="",
        log_file="",
        objects=None,
        flag_drl=False,
        flag_print_info=False
    ):
        self.flag_print_info = flag_print_info

        if self.flag_print_info:
            print("[igibson_env_jackalJaco::iGibsonEnv::__init__] START")

        ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
        ### Initialize Config Parameters
        #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] START CONFIG")
        config_igibson_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.config_igibson = parse_config(config_igibson_data)
        
        ## Set namespace
        self.init_ros_node = init_ros_node
        self.ros_node_id = ros_node_id
        robot_ns = self.config_igibson["robot_ns"]
        self.ns = "/" + robot_ns + "_" + str(ros_node_id) + "/"

        if self.ros_node_id == 0:
            self.config_mobiman = Config(log_file=log_file, drl_mode=drl_mode, flag_print_info=flag_print_info) # type: ignore
        else:
            self.config_mobiman = Config() # type: ignore
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] END CONFIG")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] START")

        # NUA NOTE: DEPRECATE ALL UNNECESSARY VARIABLES!
        ### Initialize Variables
        self.drl_mode = drl_mode
        self.flag_drl = flag_drl

        self.flag_run_sim = False
        self.init_flag = False
        #self.init_goal_flag = False
        self.init_occupancy_data_flag = False
        self.target_update_flag =  False
        self.init_update_flag0 = False
        self.init_update_flag1 = False
        self.init_update_flag2 = False
        self.callback_update_flag = False
        self.episode_done = False
        self.reached_goal = False
        self.current_episode = 0 # NUA NOTE: DEPRECATE!
        self.episode_num = 1
        self.step_num = 1
        self.total_step_num = 1
        self.total_collisions = 0
        self.total_rollover = 0
        self.total_goal = 0
        self.total_max_step = 0
        self.total_out_of_boundary = 0
        self.total_target = 0
        self.total_time_horizon = 0
        self.step_action = None
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.total_mean_episode_reward = 0.0
        #self.goal_status = Bool()
        #self.goal_status.data = False
        self.flag_action_target = False
        self.action_counter = 0
        self.observation_counter = 0
        self.mrt_ready_flag = False
        self.mpc_data_received_flag = False
        self.mpc_action_result = 0
        self.mpc_action_result_total_timestep = 0
        self.mpc_action_result_model_mode = -1
        self.mpc_action_result_com_error_norm_total = 0
        self.mpc_action_complete = False
        
        self.testing_idx = 0
        self.testing_eval_idx = 1
        self.flag_testing_done = False

        # Variables for saving OARS data
        self.data = None
        self.oars_data = {'log_file':[], 'episode_index':[], 'step_index':[], 'observation':[], 'action':[], 'reward':[], 'result':[]}
        if self.drl_mode == "testing":
            self.oars_data['testing_index'] = []
            self.oars_data['testing_state'] = []
            self.oars_data['testing_eval_index'] = []
        self.termination_reason = ''
        self.model_mode = -1
        
        self.init_robot_pose = {}
        self.robot_data = {}
        self.goal_data = {}
        #self.target_data = {}
        self.arm_data = {}
        self.occupancy_data = {}
        #self.mpc_data = {}
        self.mpc_data_msg = None
        self.manual_target_msg = None
        self.target_msg = None
        self.mobiman_goal_obs_msg = None
        self.mobiman_occupancy_obs_msg = None
        self.selfcoldistance_msg = None

        # Set initial command
        self.ctr_seq = 0
        self.cmd_seq_prev = None
        self.cmd_seq = None
        self.cmd_base_init = [0.0, 0.0]
        self.cmd_base_zeros = self.cmd_base_init
        self.cmd_base = self.cmd_base_init

        self.cmd_arm_init_j1 = 0.0
        self.cmd_arm_init_j2 = 2.9
        self.cmd_arm_init_j3 = 1.3
        self.cmd_arm_init_j4 = 4.2
        self.cmd_arm_init_j5 = 1.4
        self.cmd_arm_init_j6 = 0.0
        self.cmd_arm_init = [self.cmd_arm_init_j1, self.cmd_arm_init_j2, self.cmd_arm_init_j3, self.cmd_arm_init_j4, self.cmd_arm_init_j5, self.cmd_arm_init_j6]
        self.cmd_arm_zeros = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cmd_arm = self.cmd_arm_init

        #self.cmd = self.cmd_base + self.cmd_arm

        # Env objects
        self.objects = objects
        self.spawned_objects = []

        if self.drl_mode == "training":
            self.oar_data_file = data_folder_path + "oar_data_" + drl_mode + "_" + self.ns[1:-1] + ".csv"
        
        elif self.drl_mode == "testing":
            self.initialize_testing_domain()
            self.oar_data_file = data_folder_path + "oar_data_" + drl_mode + "_" + self.ns[1:-1] + "_" + self.config_mobiman.testing_benchmark_name + ".csv"
        
        else:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] ERROR: Invalid drl_mode!")
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
            while 1:
                continue
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] oar_data_file: " + str(self.oar_data_file))

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
        #while 1:
        #    continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] BEFORE super")
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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] num robots: " + str(len(self.robots)))

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting for mrt_ready...")
        while not self.mrt_ready_flag:
            continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG INF")
        #while 1:
        #    continue

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] END")

    '''
    DESCRIPTION: TODO...
    '''
    def init_ros_env(self, ros_node_id=0, init_flag=True):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] START")
        
        if init_flag:
            if self.flag_print_info:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] ROS entered to the chat!")
            rospy.init_node("igibson_ros_" + str(ros_node_id), anonymous=True)

            # ROS variables
            self.rospack = rospkg.RosPack()
            self.urdf_path = os.path.join(self.rospack.get_path('mobiman_simulation'), 'urdf')
            self.listener = tf.TransformListener()
            self.bridge = CvBridge()
            self.br = tf.TransformBroadcaster()
            #self.last_update_base = rospy.Time.now()
            #self.last_update_arm = rospy.Time.now()

            # Subscribers
            rospy.Subscriber(self.ns + self.config_mobiman.mpc_data_msg_name, mpc_data, self.mpc_data_callback)
            #rospy.Subscriber(self.ns + self.config_mobiman.mobiman_goal_obs_msg_name, MobimanGoalObservation, self.callback_mobiman_goal_obs)
            #rospy.Subscriber(self.ns + self.config_mobiman.mobiman_occupancy_obs_msg_name, MobimanOccupancyObservation, self.callback_mobiman_occupancy_obs)
            rospy.Subscriber(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info, self.callback_selfcoldistance)
            #rospy.Subscriber(self.ns + self.config_mobiman.target_msg_name, MarkerArray, self.callback_target)
            #rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info, self.callback_extcoldistance_base)
            #rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info, self.callback_extcoldistance_arm) # type: ignore
            #rospy.Subscriber(self.ns + self.config_mobiman.occgrid_msg_name, OccupancyGrid, self.callback_occgrid)
            #rospy.Subscriber(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray, self.callback_pointsonrobot)
            
            if self.config_mobiman.manual_target_msg_name != "":
                rospy.Subscriber(self.ns + self.config_mobiman.manual_target_msg_name, MarkerArray, self.callback_manual_target)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] BEFORE flag_run_sim: " + str(self.flag_run_sim))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] BEFORE flag_drl: " + str(self.flag_drl))
            self.flag_run_sim = True
            if not self.flag_drl:
                self.config_mobiman.goal_frame_name = rospy.get_param('gs_goal_frame_name', "")

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] AFTER flag_run_sim: " + str(self.flag_run_sim))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] DEBUG_INF")
            #while 1:
            #    continue
            
            # Publishers
            #self.image_pub = rospy.Publisher(self.ns + self.config_mobiman.rgb_image_msg_name, Image, queue_size=10)
            #self.depth_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_msg_name, Image, queue_size=10)
            #self.depth_raw_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_raw_msg_name, Image, queue_size=10)
            #self.camera_info_pub = rospy.Publisher(self.ns + self.config_mobiman.camera_info_msg_name, CameraInfo, queue_size=10)
            #self.lidar_pub = rospy.Publisher(self.ns + self.config_mobiman.lidar_msg_name, PointCloud2, queue_size=10)
            self.odom_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            #self.odom_gt_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.joint_states_pub = rospy.Publisher(self.ns + self.config_mobiman.arm_state_msg_name, JointState, queue_size=10)
            #self.goal_status_pub = rospy.Publisher(self.config_mobiman.goal_status_msg_name, Bool, queue_size=1)
            #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
            self.debug_visu_pub = rospy.Publisher(self.ns + 'debug_visu', MarkerArray, queue_size=5)
            self.model_state_pub = rospy.Publisher(self.ns + "model_states", ModelStates, queue_size=10)

            # Services
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] service_set_mrt_ready: " + str(self.ns + 'set_mrt_ready'))
            rospy.Service(self.ns + 'set_mrt_ready', setBool, self.service_set_mrt_ready)
            rospy.Service(self.ns + 'set_mpc_action_result', setMPCActionResult, self.service_set_mpc_action_result)

            self.create_objects(self.objects)

            # Timers
            rospy.Timer(rospy.Duration(0.01), self.timer_transform)
            rospy.Timer(rospy.Duration(0.01), self.timer_update)
            rospy.Timer(rospy.Duration(0.01), self.timer_sim)

            # Wait for topics
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.mobiman_goal_obs_msg_name) + "...")
            #rospy.wait_for_message(self.ns + self.config_mobiman.mobiman_goal_obs_msg_name, MobimanGoalObservation)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.mobiman_occupancy_obs_msg_name) + "...")
            #rospy.wait_for_message(self.ns + self.config_mobiman.mobiman_occupancy_obs_msg_name, MobimanOccupancyObservation)

            if self.flag_print_info:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.selfcoldistance_msg_name) + "...")
            rospy.wait_for_message(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.extcoldistance_base_msg_name) + "...")
            #rospy.wait_for_message(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.extcoldistance_arm_msg_name) + "...")
            #rospy.wait_for_message(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.pointsonrobot_msg_name) + "...")
            #rospy.wait_for_message(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray)

            if self.flag_print_info:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting callback_update_flag and init_occupancy_data_flag...")
            while (not self.callback_update_flag) or (not self.init_occupancy_data_flag):
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] callback_update_flag: " + str(self.callback_update_flag))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] init_occupancy_data_flag: " + str(self.init_occupancy_data_flag))
                continue
            
            if self.config_mobiman.manual_target_msg_name != "":
                if self.flag_print_info:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting msg: " + str(self.ns + self.config_mobiman.manual_target_msg_name) + "...")
                rospy.wait_for_message(self.ns + self.config_mobiman.manual_target_msg_name, MarkerArray)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] DEBUG_INF")
            #while 1:
            #    continue
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] END")

    '''
    DESCRIPTION: TODO...
    '''
    def create_objects(self, objects):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] START")

        for key,val in objects.items():
            if "conveyor" in key:

                if self.drl_mode == "training":
                    self.conveyor_pos = self.config_igibson["conveyor_pos"]
                    pointer = p.loadURDF(os.path.join(self.urdf_path, f"{key}.urdf"),
                                         basePosition = self.conveyor_pos[:],
                                         baseOrientation = [0, 0, 0, 1])
                    self.spawned_objects.append(pointer)

                elif self.drl_mode == "testing":
                    self.conveyor_pos = self.config_igibson["conveyor_pos"]
                    pointer = p.loadURDF(os.path.join(self.urdf_path, f"{key}.urdf"),
                                         basePosition = self.conveyor_pos[:],
                                         baseOrientation = [0, 0, 0, 1])
                    self.spawned_objects.append(pointer)

                else:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] Invalid drl_mode!")
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] DEBUG_INF_0")
                    while 1:
                        continue

            else:
                if self.drl_mode == "training":
                    temp_pos = self.conveyor_pos[:]
                    temp_pos[2] += 1.0
                    pointer = p.loadURDF(os.path.join(self.urdf_path, f"{key}.urdf"),
                                         basePosition = temp_pos,
                                         baseOrientation = [0, 0, 0, 1])
                    self.spawned_objects.append(pointer)
                
                elif self.drl_mode == "testing":
                    temp_pos = self.conveyor_pos[:]
                    temp_pos[2] += 1.0
                    pointer = p.loadURDF(os.path.join(self.urdf_path, f"{key}.urdf"),
                                         basePosition = temp_pos,
                                         baseOrientation = [0, 0, 0, 1])
                    self.spawned_objects.append(pointer)

                else:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] Invalid drl_mode!")
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] DEBUG_INF_1")
                    while 1:
                        continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] conveyor_pos: " + str(self.conveyor_pos))

        self.randomize_domain()

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::create_objects] END")

    '''
    DESCRIPTION: TODO...
    '''
    def randomize_env(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] START")

        for idx, (key,val) in enumerate(self.objects.items()):
            if "conveyor" in key:

                if self.drl_mode == "training":
                    p.resetBasePositionAndOrientation(self.spawned_objects[idx], 
                                                      posObj=self.conveyor_pos[:], 
                                                      ornObj=[0, 0, 0, 1])
                
                elif self.drl_mode == "testing":
                    p.resetBasePositionAndOrientation(self.spawned_objects[idx], 
                                                      posObj=self.conveyor_pos[:], 
                                                      ornObj=[0, 0, 0, 1])
                
                else:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] Invalid drl_mode!")
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] DEBUG_INF_0")
                    while 1:
                        continue
            else:
                
                if self.drl_mode == "training":
                    shift_x = random.uniform(self.config_mobiman.goal_range_min_x, self.config_mobiman.goal_range_max_x)
                    shift_y = random.uniform(self.config_mobiman.goal_range_min_y, self.config_mobiman.goal_range_max_y)
                    shift_z = random.uniform(self.config_mobiman.goal_range_min_z, self.config_mobiman.goal_range_max_z)
                
                elif self.drl_mode == "testing":
                    shift_x = self.testing_samples[self.testing_idx][3]
                    shift_y = self.testing_samples[self.testing_idx][4]
                    shift_z = self.testing_samples[self.testing_idx][5]

                else:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] Invalid drl_mode!")
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] DEBUG_INF_1")
                    while 1:
                        continue
                
                '''
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] DEBUG_WARNING: MANUALLY SET MODEL MODE! CHANGE IT BACK ASAP!!!")
                if self.ros_node_id == 0:
                    shift_x = -0.5
                    shift_y = 0.0
                    shift_z = 0.6
                else:
                    shift_x = 0.5
                    shift_y = 0.0
                    shift_z = 0.6
                '''

                if self.flag_print_info:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] conveyor_pos: " + str(self.conveyor_pos))
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] shift_x: " + str(shift_x))
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] shift_y: " + str(shift_y))
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] shift_z: " + str(shift_z))

                temp_pos = self.conveyor_pos[:]
                temp_pos[0] += shift_x
                temp_pos[1] += shift_y
                temp_pos[2] = shift_z
                p.resetBasePositionAndOrientation(self.spawned_objects[idx], 
                                                  posObj=temp_pos, 
                                                  ornObj=[0, 0, 0, 1])
                
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] temp_pos: " + str(temp_pos))

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_env] END")

    '''
    DESCRIPTION: TODO...
    '''
    def timer_transform(self, timer):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_transform] START")
        try:
            model_state_msg = ModelStates()
            for obj, dict in zip(self.spawned_objects, self.objects.items()):
                # self.br.sendTransform(obj.get_position(), obj.get_orientation(), rospy.Time.now(), f'{self.ns}{dict[0]}', 'world')
                model_state_msg.name.append(dict[0])
                POSE, ORIENTATION = p.getBasePositionAndOrientation(obj)
                pose = Pose() 
                x,y,z = POSE
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                x,y,z,w = ORIENTATION
                pose.orientation.x = x
                pose.orientation.y = y
                pose.orientation.z = z
                pose.orientation.w = w
                model_state_msg.pose.append(pose)
            self.model_state_pub.publish(model_state_msg)
        except Exception as e:
            pass     
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_transform] END")

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_base_callback(self, data):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::cmd_base_callback] INCOMING")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::cmd_base_callback] DEBUG_INF")
        while 1:
            continue
        self.cmd_base = [data.linear.x, data.angular.z]
        self.last_update_base = rospy.Time.now()

        '''
        if self.mrt_ready_flag:
            self.cmd_base = [data.linear.x, data.angular.z]
            self.last_update_base = rospy.Time.now()
        else:
            self.cmd_base = self.cmd_base_init
        '''

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_arm_callback(self, data):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::cmd_arm_callback] INCOMING")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::cmd_arm_callback] DEBUG_INF")
        while 1:
            continue
        self.cmd_arm = list(data.points[0].positions)
        
        '''
        if self.mrt_ready_flag:
            #joint_names = data.joint_names
            self.cmd_arm = list(data.points[0].positions)
        else:
            self.cmd_arm = self.cmd_arm_init
        '''

    '''
    DESCRIPTION: TODO...
    '''
    def mpc_data_callback(self, data):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::mpc_data_callback] INCOMING")        
        self.mpc_data_msg = data

    '''
    DESCRIPTION: TODO...
    '''
    def callback_modelmode(self, msg):
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_modelmode] INCOMING")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_modelmode] DEBUG_INF")
        while 1:
            continue
        self.model_mode = msg.data
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_modelmode] model_mode: " + str(self.model_mode))

    '''
    DESCRIPTION: TODO...
    '''
    def callback_manual_target(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_manual_target] INCOMING")
        self.manual_target_msg = msg
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_manual_target] DEBUG_INF")
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def callback_target(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_target] INCOMING")
        self.target_msg = msg
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_target] DEBUG_INF")
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def callback_occgrid(self, msg):
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_occgrid] INCOMING")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_occgrid] DEBUG_INF")
        while 1:
            continue
        self.occgrid_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_mobiman_goal_obs(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_mobiman_goal_obs] INCOMING")
        self.mobiman_goal_obs_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_mobiman_occupancy_obs(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_mobiman_occupancy_obs] INCOMING")
        self.mobiman_occupancy_obs_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_selfcoldistance(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_selfcoldistance] INCOMING")
        self.selfcoldistance_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_base(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_base] INCOMING")
        self.extcoldistance_base_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_arm(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_arm] INCOMING")
        self.extcoldistance_arm_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_pointsonrobot(self, msg):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::callback_pointsonrobot] INCOMING")
        self.pointsonrobot_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def timer_update(self, event):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] START")

        self.update_robot_data()
        self.update_arm_data()
        self.update_ros_topics()
        #self.update_target_data()

        goal_frame_name = self.ns + self.config_mobiman.goal_frame_name
        robot_frame_name = self.ns + self.config_mobiman.robot_frame_name
        ee_frame_name = self.ns + self.config_mobiman.ee_frame_name

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] world_frame_name: " + str(self.config_mobiman.world_frame_name))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] goal_frame_name: " + str(goal_frame_name))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] robot_frame_name: " + str(robot_frame_name))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] ee_frame_name: " + str(ee_frame_name))

        try:
            (trans_goal_wrt_world, rot_goal_wrt_world) = self.listener.lookupTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=goal_frame_name, time=rospy.Time(0))
            self.update_goal_data(trans_goal_wrt_world, rot_goal_wrt_world)
            self.init_update_flag0 = True

        except Exception as e0:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] ERROR e0: " + str(e0))
            ...

        try:
            goal_frame_name = self.ns + self.config_mobiman.goal_frame_name
            (trans_goal_wrt_robot, rot_goal_wrt_robot) = self.listener.lookupTransform(target_frame=robot_frame_name, source_frame=goal_frame_name, time=rospy.Time(0))
            self.update_goal_data_wrt_robot(trans_goal_wrt_robot, rot_goal_wrt_robot)
            self.init_update_flag1 = True

        except Exception as e1:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] ERROR e1: " + str(e1))
            ...

        try:
            goal_frame_name = self.ns + self.config_mobiman.goal_frame_name
            (trans_goal_wrt_ee, rot_goal_wrt_ee) = self.listener.lookupTransform(target_frame=ee_frame_name, source_frame=goal_frame_name, time=rospy.Time(0))
            self.update_goal_data_wrt_ee(trans_goal_wrt_ee, rot_goal_wrt_ee)
            self.init_update_flag2 = True

        except Exception as e2:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] ERROR e2: " + str(e2))
            ...

        if self.init_update_flag0 and self.init_update_flag1 and self.init_update_flag2:
            self.callback_update_flag = True

        try:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] occupancy_frame_names: " + str(self.config_mobiman.occupancy_frame_names))
            for i, occ in enumerate(self.config_mobiman.occupancy_frame_names):
                occ_frame_name = self.ns + occ
                robot_frame_name = self.ns + self.config_mobiman.robot_frame_name
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] occ_withNS: " + str(occ_withNS))
                (trans_occ_wrt_robot, rot_occ_wrt_robot) = self.listener.lookupTransform(target_frame=robot_frame_name, source_frame=occ_frame_name, time=rospy.Time(0))                
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] occ: " + str(occ))
                self.update_occupancy_data(occ, trans_occ_wrt_robot, rot_occ_wrt_robot)
            self.init_occupancy_data_flag = True

        except Exception as e3:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] ERROR e3: " + str(e3))
            ...

        '''
        trans_robot_wrt_world = tf.transformations.translation_matrix([self.robot_data["x"], self.robot_data["y"], self.robot_data["z"]])
        quat_robot_wrt_world = tf.transformations.quaternion_matrix([self.robot_data["qx"], self.robot_data["qy"], self.robot_data["qz"], self.robot_data["qw"]])
        tf_robot_wrt_world = tf.transformations.concatenate_matrices(trans_robot_wrt_world, quat_robot_wrt_world)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] CALCULATED trans_robot_wrt_world")
        #print(trans_robot_wrt_world)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] CALCULATED tf_robot_wrt_world")
        #print(tf_robot_wrt_world)
        #print("------")

        trans_ee_wrt_world = tf.transformations.translation_matrix([self.arm_data["x"], self.arm_data["y"], self.arm_data["z"]])
        quat_ee_wrt_world = tf.transformations.quaternion_matrix([self.arm_data["qx"], self.arm_data["qy"], self.arm_data["qz"], self.arm_data["qw"]])
        tf_ee_wrt_world = tf.transformations.concatenate_matrices(trans_ee_wrt_world, quat_ee_wrt_world)
        
        trans_goal_wrt_world = tf.transformations.translation_matrix(self.trans_goal_wrt_world)
        quat_goal_wrt_world = tf.transformations.quaternion_matrix(self.rot_goal_wrt_world)
        tf_goal_wrt_world = tf.transformations.concatenate_matrices(trans_goal_wrt_world, quat_goal_wrt_world)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] CALCULATED trans_goal_wrt_world")
        #print(trans_goal_wrt_world)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] CALCULATED tf_goal_wrt_world")
        #print(tf_goal_wrt_world)
        #print("------")

        # Calculate the transformation from end effector wrt base
        tf_world_wrt_robot = tf.transformations.inverse_matrix(tf_robot_wrt_world)
        transform_goal_wrt_robot = tf.transformations.concatenate_matrices(tf_world_wrt_robot, tf_goal_wrt_world)
        
        tf_world_wrt_ee = tf.transformations.inverse_matrix(tf_ee_wrt_world)
        transform_goal_wrt_ee = tf.transformations.concatenate_matrices(tf_world_wrt_ee, tf_goal_wrt_world)

        self.trans_goal_wrt_robot = tf.transformations.translation_from_matrix(transform_goal_wrt_robot)
        self.rot_goal_wrt_robot = tf.transformations.quaternion_from_matrix(transform_goal_wrt_robot)
        
        self.trans_goal_wrt_ee = tf.transformations.translation_from_matrix(transform_goal_wrt_ee)
        self.rot_goal_wrt_ee = tf.transformations.quaternion_from_matrix(transform_goal_wrt_ee)
        
        self.callback_update_flag = True

        self.update_goal_data()
        self.update_goal_data_wrt_robot()
        self.update_goal_data_wrt_ee()
        '''
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_update] END")

    '''
    DESCRIPTION: TODO...
    '''
    def timer_sim(self, event):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_sim] START")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_sim] flag_run_sim: " + str(self.flag_run_sim))
        if self.flag_run_sim:

            if self.mpc_data_msg:
                mpc_data_msg = self.mpc_data_msg
                self.mpc_data_received_flag = True

                self.cmd_seq = mpc_data_msg.seq
                self.model_mode = mpc_data_msg.model_mode
                self.input_state = mpc_data_msg.input_state

                if self.model_mode == 0:
                    self.cmd_base = list(mpc_data_msg.cmd)
                elif self.model_mode == 1:
                    self.cmd_base = self.cmd_base_zeros
                    self.cmd_arm = list(mpc_data_msg.cmd)
                elif self.model_mode == 2:
                    cmd_wb = list(mpc_data_msg.cmd)
                    self.cmd_base = cmd_wb[:len(self.cmd_base)]
                    self.cmd_arm = cmd_wb[len(self.cmd_base):]
                else:
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::mpc_data_callback] ERROR: Invalid model_mode: " + str(mpc_data_msg.model_mode))
                    self.cmd_base = self.cmd_base_zeros
         
            if self.cmd_seq_prev == self.cmd_seq:
                self.ctr_seq = self.ctr_seq + 1
            else:
                self.ctr_seq = 0

            ### NUA NOTE: SET THE PARAM IN CONFIG!  
            if self.mpc_data_received_flag and self.ctr_seq > 10:
                self.cmd_base = self.cmd_base_zeros
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::mpc_data_callback] ERROR: SAME COMMAND OVER 10 TIMES!")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::mpc_data_callback] cmd_seq: " + str(self.cmd_seq))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::mpc_data_callback] cmd_seq_prev: " + str(self.cmd_seq_prev))
                  
            cmd = self.cmd_base + self.cmd_arm
            #cmd = self.cmd_base_init + self.cmd_arm_init
            self.robots[0].apply_action(cmd)
            self.simulator_step()
            self.cmd_seq_prev = self.cmd_seq
    
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::timer_sim] END")

    '''
    DESCRIPTION: TODO...
    '''
    def client_movebase(self):
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_movebase] START")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_movebase] DEBUG_INF")
        while 1:
            continue
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

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_movebase] END")

    '''
    DESCRIPTION: TODO...
    '''
    def client_set_action_drl(self, action):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Waiting for service set_action_drl...")
        
        try:
            if self.config_mobiman.action_type == 0:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] DISCRETE ACTION")
                
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] DEBUG_INF")
                #while 1:
                #    continue
                
                set_action_drl_service_name = self.ns + 'set_discrete_action_drl_mrt'
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] set_action_drl_service_name: " + str(set_action_drl_service_name))
                rospy.wait_for_service(set_action_drl_service_name)
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Received service set_action_drl!")

                srv_set_discrete_action_drl = rospy.ServiceProxy(set_action_drl_service_name, setDiscreteActionDRL)  
                success = srv_set_discrete_action_drl(self.total_step_num, action, self.config_mobiman.action_time_horizon).success
            else:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] CONTINUOUS ACTION")
                set_action_drl_service_name = self.ns + 'set_continuous_action_drl_mrt'
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] set_action_drl_service_name: " + str(set_action_drl_service_name))
                rospy.wait_for_service(set_action_drl_service_name)
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Received service set_action_drl!")

                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] action: "  + str(action))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] action_time_horizon: "  + str(self.config_mobiman.action_time_horizon))
                srv_set_continuous_action_drl = rospy.ServiceProxy(set_action_drl_service_name, setContinuousActionDRL)
                success = srv_set_continuous_action_drl(self.total_step_num, action, self.config_mobiman.action_time_horizon).success
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] CONTINUOUS ACTION SENT!")
            '''
            if(success):
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] Updated action: " + str(action))
            else:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] goal_pose is NOT updated!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] ERROR: set_action_drl is NOT successful!")
            '''
            return success

        except rospy.ServiceException as e:  
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::client_set_action_drl] ERROR: Service call failed: %s"%e)
            return False
        
    '''
    DESCRIPTION: TODO...
    '''
    def service_set_mrt_ready(self, req):
        self.mrt_ready_flag = req.val
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
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] START")
        self.mpc_action_result = req.action_result
            
        if self.mpc_action_result == 0:
            self.termination_reason = 'out_of_boundary'
            self.total_out_of_boundary += 1
            self.episode_done = True

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
        else:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] mpc_action_result: " + str(self.mpc_action_result))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] ERROR: INVALID MPC ACTION RESULT !!!")

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] DEBUG INF")
            while 1:
                continue
        
        self.mpc_action_result_total_timestep = req.total_timestep
        self.mpc_action_result_model_mode = req.model_mode
        self.mpc_action_result_com_error_norm_total = req.com_error_norm_total
        self.mpc_action_complete = True
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] termination_reason: " + str(self.termination_reason))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] mpc_action_result: " + str(self.mpc_action_result))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::service_set_mpc_action_result] END")
        return setMPCActionResultResponse(True)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_mobiman_goal_obs_config(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_mobiman_goal_obs_config] START")
        
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_mobiman_goal_obs_config] DEBUG_INF")
        while 1:
            continue

        n_mobiman_goal_obs = int(len(self.mobiman_goal_obs_msg.obs))
        mobiman_goal_obs_frame_id = self.mobiman_goal_obs_msg.header.frame_id
        mobiman_goal_obs_dim_dt = self.mobiman_goal_obs_msg.dim_dt
        self.config_mobiman.set_mobiman_goal_obs_config(n_mobiman_goal_obs, mobiman_goal_obs_frame_id, mobiman_goal_obs_dim_dt)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_mobiman_goal_obs_config] END")

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_mobiman_occupancy_obs_config(self):

        #self.occupancy_data["names"] = self.config_mobiman.occupancy_frame_names
        #self.occupancy_data["pos"] =

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_mobiman_occupancy_obs_config] DEBUG_INF")
        #while 1:
        #    continue

        '''
        n_mobiman_occupancy_obs = int(len(self.mobiman_occupancy_obs_msg.obs))
        mobiman_occupancy_obs_frame_id = self.mobiman_occupancy_obs_msg.header.frame_id
        mobiman_occupancy_obs_dim_dt = self.mobiman_occupancy_obs_msg.dim_dt
        self.config_mobiman.set_mobiman_occupancy_obs_config(n_mobiman_occupancy_obs, mobiman_occupancy_obs_frame_id, mobiman_occupancy_obs_dim_dt)
        '''

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
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] START " )

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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] x: " + str(self.robot_data["x"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] y: " + str(self.robot_data["y"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] z: " + str(self.robot_data["z"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qx: " + str(self.robot_data["qx"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qy: " + str(self.robot_data["qy"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qz: " + str(self.robot_data["qz"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qw: " + str(self.robot_data["qw"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_robot_data] END" )

    '''
    DESCRIPTION: TODO... Update arm data
    '''
    def update_arm_data(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] START " )
        
        #link_names = self.robots[0].get_link_names()
        #base_pos, base_quat = self.robots[0].get_base_link_position_orientation()
        ee_pos, ee_quat = self.robots[0].get_link_position_orientation(self.config_mobiman.ee_frame_name)
        ee_rpy = self.robots[0].get_link_rpy(self.config_mobiman.ee_frame_name)
        
        '''
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] link_names: " )
        print(link_names)
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_frame_name: " + str(self.config_mobiman.ee_frame_name))
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_pos: " + str(ee_pos))
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_quat: " + str(ee_quat))

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_pos: " + str(base_pos))
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_quat: " + str(base_quat))

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] DEBUG_INF" )
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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] x: " + str(self.arm_data["x"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] y: " + str(self.arm_data["y"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] z: " + str(self.arm_data["z"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qx: " + str(self.arm_data["qx"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qy: " + str(self.arm_data["qy"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qz: " + str(self.arm_data["qz"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qw: " + str(self.arm_data["qw"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] joint_names: ")
        #print(joint_names)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] joint_pos: ")
        #print(self.arm_data["joint_pos"])
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] END" )

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_arm_data] DEBUG_INF" )
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data(self, translation_wrt_world, rotation_wrt_world):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] START")
        
        #translation_wrt_world = self.trans_goal_wrt_world
        #rotation_wrt_world = self.rot_goal_wrt_world
        
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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] x: " + str(self.goal_data["x"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] y: " + str(self.goal_data["y"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] z: " + str(self.goal_data["z"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qx: " + str(self.goal_data["qx"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qy: " + str(self.goal_data["qy"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qz: " + str(self.goal_data["qz"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qw: " + str(self.goal_data["qw"]))

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_robot(self, translation_wrt_robot, rotation_wrt_robot):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] START")

        #translation_wrt_robot = self.trans_goal_wrt_robot
        #rotation_wrt_robot = self.rot_goal_wrt_robot

        self.goal_data["x_wrt_robot"] = translation_wrt_robot[0] # type: ignore
        self.goal_data["y_wrt_robot"] = translation_wrt_robot[1] # type: ignore
        self.goal_data["z_wrt_robot"] = translation_wrt_robot[2] # type: ignore
        self.goal_data["qx_wrt_robot"] = rotation_wrt_robot[0] # type: ignore
        self.goal_data["qy_wrt_robot"] = rotation_wrt_robot[1] # type: ignore
        self.goal_data["qz_wrt_robot"] = rotation_wrt_robot[2] # type: ignore
        self.goal_data["qw_wrt_robot"] = rotation_wrt_robot[3] # type: ignore

        #q = Quaternion(rotation_wrt_robot[3], rotation_wrt_robot[0], rotation_wrt_robot[1], rotation_wrt_robot[2]) # type: ignore
        #e = q.to_euler(degrees=False)
        #self.goal_data["roll_wrt_robot"] = e[0] # type: ignore
        #self.goal_data["pitch_wrt_robot"] = e[1] # type: ignore
        #self.goal_data["yaw_wrt_robot"] = e[2] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] x_wrt_robot: " + str(self.goal_data["x_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] y_wrt_robot: " + str(self.goal_data["y_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] z_wrt_robot: " + str(self.goal_data["z_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qx_wrt_robot: " + str(self.goal_data["qx_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qy_wrt_robot: " + str(self.goal_data["qy_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qz_wrt_robot: " + str(self.goal_data["qz_wrt_robot"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qw_wrt_robot: " + str(self.goal_data["qw_wrt_robot"]))

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_ee(self, translation_wrt_ee, rotation_wrt_ee):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] START")

        #translation_wrt_ee = self.trans_goal_wrt_ee
        #rotation_wrt_ee = self.rot_goal_wrt_ee

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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] x_wrt_ee: " + str(self.goal_data["x_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] y_wrt_ee: " + str(self.goal_data["y_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] z_wrt_ee: " + str(self.goal_data["z_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qx_wrt_ee: " + str(self.goal_data["qx_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qy_wrt_ee: " + str(self.goal_data["qy_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qz_wrt_ee: " + str(self.goal_data["qz_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qw_wrt_ee: " + str(self.goal_data["qw_wrt_ee"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_target_data(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] START")
        
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] DEBUG_INF")
        while 1:
            continue

        if self.target_msg:
            target_msg = self.target_msg

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] x: " + str(self.target_msg.markers[0].pose.position.x))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] y: " + str(self.target_msg.markers[0].pose.position.y))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] z: " + str(self.target_msg.markers[0].pose.position.z))
            
            q = Quaternion(target_msg.markers[0].pose.orientation.w, target_msg.markers[0].pose.orientation.x, target_msg.markers[0].pose.orientation.y, target_msg.markers[0].pose.orientation.z) # type: ignore
            e = q.to_euler(degrees=False)

            ## NUA TODO: Generalize to multiple target points!
            self.target_data["x"] = target_msg.markers[0].pose.position.x
            self.target_data["y"] = target_msg.markers[0].pose.position.y
            self.target_data["z"] = target_msg.markers[0].pose.position.z

            self.target_data["qx"] = target_msg.markers[0].pose.orientation.x
            self.target_data["qy"] = target_msg.markers[0].pose.orientation.y
            self.target_data["qz"] = target_msg.markers[0].pose.orientation.z
            self.target_data["qw"] = target_msg.markers[0].pose.orientation.w

            self.target_data["roll"] = e[0] # type: ignore
            self.target_data["pitch"] = e[1] # type: ignore
            self.target_data["yaw"] = e[2] # type: ignore

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] UPDATED!")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] x: " + str(self.target_data["x"]))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] y: " + str(self.target_data["y"]))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] z: " + str(self.target_data["z"]))

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] roll: " + str(self.target_data["roll"]))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] pitch: " + str(self.target_data["pitch"]))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] yaw: " + str(self.target_data["yaw"]))

            self.target_update_flag = True

            '''
            p = Point()
            p.x = self.target_data["x"]
            p.y = self.target_data["y"]
            p.z = self.target_data["z"]
            debug_point_data = [p]
            self.publish_debug_visu(debug_point_data)
            '''

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_target_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_occupancy_data(self, name, trans_occ_wrt_robot, rot_occ_wrt_robot):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_occupancy_data] START")
        self.occupancy_data[name] = [trans_occ_wrt_robot[0], trans_occ_wrt_robot[1], trans_occ_wrt_robot[2]]
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_occupancy_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_ros_topics(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] START")

        ## Odom
        odom = [np.array(self.robots[0].get_position()),
                np.array(self.robots[0].get_rpy())]

        self.br.sendTransform(
            (odom[0][0], odom[0][1], odom[0][2]),
            tf.transformations.quaternion_from_euler(odom[-1][0], odom[-1][1], odom[-1][2]), # type: ignore
            rospy.Time.now(),
            self.ns + "base_link",
            self.ns + "odom")

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = self.ns + "odom"
        odom_msg.child_frame_id = self.ns + "base_link"

        odom_msg.pose.pose.position.x = odom[0][0]
        odom_msg.pose.pose.position.y = odom[0][1]
        odom_msg.pose.pose.position.z = odom[0][2]
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
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_ros_topics2] odom_msg: " + str(odom_msg))

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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_observation(self, check_flag=False):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] START")

        if self.config_mobiman.observation_space_type == "mobiman_FC":
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] mobiman_FC")

            # Get OccGrid array observation
            #obs_occgrid = self.get_obs_occgrid()

            # Update goal observation
            obs_mobiman_goal, obs_mobiman_goal_check = self.get_obs_mobiman_goal(check_flag=check_flag)
            if not obs_mobiman_goal_check:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] ERROR: obs_mobiman_goal_check failed!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_mobiman_goal shape: " + str(obs_mobiman_goal.shape))
                print(obs_mobiman_goal)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] DEBUG INF")
                while 1:
                    continue

            # Update occupancy observation
            obs_mobiman_occupancy, obs_mobiman_occupancy_check = self.get_obs_mobiman_occupancy(check_flag=check_flag)
            if not obs_mobiman_occupancy_check:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] ERROR: obs_mobiman_occupancy_check failed!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_mobiman_occupancy shape: " + str(obs_mobiman_occupancy.shape))
                print(obs_mobiman_occupancy)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] DEBUG INF")
                while 1:
                    continue

            # Get collision sphere distance observation
            obs_selfcoldistance, obs_selfcoldistance_check = self.get_obs_selfcoldistance(check_flag=check_flag)
            if not obs_selfcoldistance_check:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] ERROR: obs_selfcoldistance_check failed!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_selfcoldistance shape: " + str(obs_selfcoldistance.shape))
                print(obs_selfcoldistance)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] DEBUG INF")
                while 1:
                    continue

            #obs_extcoldistance_base = self.get_obs_extcoldistance_base()
            #obs_extcoldistance_arm = self.get_obs_extcoldistance_arm()

            # Update base velocity observation
            obs_base_velo, obs_base_velo_check = self.get_obs_base_velo(check_flag=check_flag)
            if not obs_base_velo_check:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] ERROR: obs_base_velo_check failed!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_base_velo shape: " + str(obs_base_velo.shape))
                print(obs_base_velo)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] DEBUG INF")
                while 1:
                    continue

            # Update arm joint observation
            obs_arm_pos, obs_arm_velo, obs_arm_state_check = self.get_obs_armstate(check_flag=check_flag)
            if not obs_arm_state_check:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] ERROR: obs_arm_state_check failed!")
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_arm_pos shape: " + str(obs_arm_pos.shape))
                print(obs_arm_pos)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_arm_velo shape: " + str(obs_arm_velo.shape))
                print(obs_arm_velo)
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] DEBUG INF")
                while 1:
                    continue
            
            '''
            p = Point()
            p.x = obs_mobiman_goal[0]
            p.y = obs_mobiman_goal[1]
            p.z = obs_mobiman_goal[2]
            debug_point_data = [p]
            robot_frame_name = self.ns + self.config_mobiman.robot_frame_name
            self.publish_debug_visu(debug_point_data, frame_name=robot_frame_name[1:])
            '''
            
            # Update observation
            self.obs = np.concatenate((obs_mobiman_goal, obs_mobiman_occupancy, obs_selfcoldistance, obs_base_velo, obs_arm_pos, obs_arm_velo), axis=0)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_mobiman_goal shape: " + str(obs_mobiman_goal.shape))
            #print(obs_mobiman_goal)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_mobiman_occupancy shape: " + str(obs_mobiman_occupancy.shape))
            #print(obs_mobiman_occupancy)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_selfcoldistance shape: " + str(obs_selfcoldistance.shape))
            #print(obs_selfcoldistance)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_base_velo shape: " + str(obs_base_velo.shape))
            #print(obs_base_velo)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_arm_pos shape: " + str(obs_arm_pos.shape))
            #print(obs_arm_pos)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_arm_velo shape: " + str(obs_arm_velo.shape))
            #print(obs_arm_velo)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs: " + str(self.obs.shape))
            #print(self.obs)

        elif self.config_mobiman.observation_space_type == "mobiman_2DCNN_FC":

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] NEEDS REVIEW: DEBUG INF")
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
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data occgrid_image shape: " + str(self.obs_data["occgrid_image"].shape))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistancedist shape: " + str(self.obs_data["extcoldistancedist"].shape))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            ##print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image: ")
            ##print(obs_space_laser_image[0, 65:75])
            ##print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target dist: " + str(obs_target[0,0]))
            ##print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target angle: " + str(obs_target[0,1] * 180 / math.pi))
            ##print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] previous_action: " + str(self.previous_action))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_occgrid_image shape: " + str(obs_occgrid_image.shape))
            ##print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image type: " + str(type(obs_space_laser_image)))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_occgrid_image shape: " + str(obs_space_occgrid_image.shape))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_coldistance_goal shape: " + str(obs_space_coldistance_goal.shape))
            #print("****************")

            self.obs["occgrid_image"] = obs_space_occgrid_image
            self.obs["coldistance_goal"] = obs_space_coldistance_goal
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::update_observation] END")

    '''
    DESCRIPTION: TODO...
    '''
    def take_action(self, action):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] START")
        
        if self.config_mobiman.action_type == 1 and self.config_mobiman.manual_target_msg_name != "":
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0#self.manual_target_msg.markers[0].pose.position.x
            action[3] = 0.0#self.manual_target_msg.markers[0].pose.position.y
            action[4] = 0.0#self.manual_target_msg.markers[0].pose.position.z

            q = Quaternion(self.manual_target_msg.markers[0].pose.orientation.w, 
                           self.manual_target_msg.markers[0].pose.orientation.x, 
                           self.manual_target_msg.markers[0].pose.orientation.y, 
                           self.manual_target_msg.markers[0].pose.orientation.z) # type: ignore
            e = q.to_euler(degrees=False)

            action[5] = 0.0#e[0]
            action[6] = 0.0#e[1]
            action[7] = 0.0#e[2]
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] manual_target_msg_name: " + str(self.config_mobiman.manual_target_msg_name))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] DEBUG_WARNING: MANUALLY SET ACTION: "+ str(action) + " CHANGE IT BACK ASAP!!!")
        
        elif self.config_mobiman.action_type == 0 and self.config_mobiman.manual_target_msg_name != "":
            action = 15
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] manual_target_msg_name: " + str(self.config_mobiman.manual_target_msg_name))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] DEBUG_WARNING: MANUALLY SET ACTION: "+ str(action) + " CHANGE IT BACK ASAP!!!")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] drl_mode: " + str(self.drl_mode))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] testing_benchmark_name: " + str(self.config_mobiman.testing_benchmark_name))

        if self.config_mobiman.action_type == 1 and self.drl_mode == "testing" and self.config_mobiman.testing_benchmark_name == "ocs2wb":
            action[0] = 1.0     # Model mode: Whole-body
            action[1] = 1.0     # Target mode: Goal

        self.step_action = action
        self.current_step += 1

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] total_step_num: " + str(self.total_step_num))

        '''
        if action[1] <= 0.5:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] TARGET:")
        else:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] GOAL:")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] target: " + str(action[3:]))
        '''

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting for mrt_ready...")
        while not self.mrt_ready_flag:
            continue
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] Recieved mrt_ready!")

        self.previous_base_distance2goal_2D = self.get_base_distance2goal_2D()
        self.previous_arm_distance2goal_3D = self.get_arm_distance2goal_3D()
        self.previous_base_wrt_world_2D = [self.robot_data["x"], self.robot_data["y"], self.robot_data["yaw"]]
        self.previous_goal_wrt_world_2D = [self.goal_data["x"], self.goal_data["y"], self.goal_data["yaw"]]

        '''
        desired_robot_wrt_world_2D = self.calculate_middle_point(self.previous_base_wrt_world_2D[:-1], 
                                                                 self.previous_goal_wrt_world_2D[:-1], 
                                                                 self.config_mobiman.reward_step_target_intermediate_point_scale)
        
        input_pose = PoseStamped()
        robot_frame_name = self.ns + self.config_mobiman.robot_frame_name
        input_pose.header.frame_id = "world" # Input pose frame
        input_pose.pose.position.x = desired_robot_wrt_world_2D[0]
        input_pose.pose.position.y = desired_robot_wrt_world_2D[1]
        input_pose.pose.position.z = 0.0
        transformed_pose = self.transform_pose(input_pose, robot_frame_name[1:])

        action[2] = transformed_pose.pose.position.x
        action[3] = transformed_pose.pose.position.y
        '''

        if self.config_mobiman.action_type == 0 and action < self.config_mobiman.n_discrete_action - 3:
            self.flag_action_target = True
        
        elif self.config_mobiman.action_type == 1 and action[1] <= 0.5:
            self.flag_action_target = True
        else:
            self.flag_action_target = False

        # Run Action Server
        success = self.client_set_action_drl(action)

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] Waiting mpc_action_complete for " + str(self.config_mobiman.action_time_horizon) + " sec... ")
        time_start = time.time()
        while not self.mpc_action_complete:
            continue

        time_end = time.time()
        self.dt_action = round(time_end - time_start, 2)
        self.mpc_action_complete = False

        self.current_base_distance2goal_2D = self.get_base_distance2goal_2D()
        self.current_arm_distance2goal_3D = self.get_arm_distance2goal_3D()
        self.current_base_wrt_world_2D = [self.robot_data["x"], self.robot_data["y"], self.robot_data["yaw"]]

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] Action completed in " + str(self.dt_action) + " sec!")
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] mpc_action_result: " + str(self.mpc_action_result))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] termination_reason: " + str(self.termination_reason))

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] DEBUG INF")
        #while 1:
        #    continue
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::take_action] END")

    '''
    DESCRIPTION: TODO...
    '''
    def is_done(self, observations):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] START")
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] total_step_num: " + str(self.total_step_num))

        flag_truncated = False
        if self.step_num >= self.config_mobiman.max_episode_steps: # type: ignore
            self.termination_reason = 'max_step'
            self.total_max_step += 1
            self.episode_done = True
            flag_truncated = True
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] Too late...")

        if self.episode_done and (not self.reached_goal):
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
        elif self.episode_done and self.reached_goal:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
        else:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] termination_reason: " + self.termination_reason) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_is_done] END")
        return self.episode_done, flag_truncated

    '''
    DESCRIPTION: TODO...
    '''
    def compute_reward(self, observations, done):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::_compute_reward] START")

        if self.episode_done and (not self.reached_goal):
            # 0: Out of Map Boundary
            # 1: Collision
            # 2: Rollover
            # 3: Goal reached
            # 4: Target reached
            # 5: Time-horizon reached
            
            if self.termination_reason == 'out_of_boundary':
                self.step_reward = self.config_mobiman.reward_terminal_out_of_boundary
            elif self.termination_reason == 'collision':
                self.step_reward = self.config_mobiman.reward_terminal_collision
            elif self.termination_reason == 'rollover':
                self.step_reward = self.config_mobiman.reward_terminal_rollover
            elif self.termination_reason == 'max_step':
                self.step_reward = self.config_mobiman.reward_terminal_max_step
            else:
                self.step_reward = self.config_mobiman.reward_terminal_collision
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] Invalid self.termination_reason: " + str(self.termination_reason) + "!")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG INF_0")
                #while 1:
                #    continue

            #self.goal_status.data = False
            #self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            #self.training_data.append([self.episode_reward])

        elif self.episode_done and self.reached_goal:

            ###print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] GOOOOOOOOOAAAAAAAALLLLLLLLLLLLLLLLLL!!!")
            self.step_reward = self.config_mobiman.reward_terminal_goal
            #self.goal_status.data = True
            #self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            #self.training_data.append([self.episode_reward])

        else:
            # Step Reward 1: model mode
            reward_step_mode = 0

            if self.model_mode == 0:
                reward_step_mode = self.config_mobiman.reward_step_mode0

            elif self.model_mode == 1:
                reward_step_mode = self.config_mobiman.reward_step_mode1

            elif self.model_mode == 2:
                reward_step_mode = self.config_mobiman.reward_step_mode2

            else:
                reward_step_mode = self.config_mobiman.reward_step_mode2
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] Invalid self.model_mode (reward_step_mode): " + str(self.model_mode) + "!")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG_INF_1")
                #while 1:
                #    continue
            weighted_reward_step_mode = self.config_mobiman.alpha_step_mode * reward_step_mode # type: ignore

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] reward_step_mode: " + str(reward_step_mode))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] weighted_reward_step_goal: " + str(weighted_reward_step_goal))

            # Step Reward 2: goal
            reward_step_goal = 0.0

            if self.model_mode == 0:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] BASE MOTION")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] current_base_distance2goal_2D: " + str(self.current_base_distance2goal_2D))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] previous_base_distance2goal_2D: " + str(self.previous_base_distance2goal_2D))
                reward_step_goal = self.get_reward_step_goal(curr_dist2goal=self.current_base_distance2goal_2D, prev_dist2goal=self.previous_base_distance2goal_2D)

            elif self.model_mode == 1 or self.model_mode == 2:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] ARM OR WHOLE-BODY MOTION")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] previous_arm_distance2goal_3D: " + str(self.previous_arm_distance2goal_3D))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] current_arm_distance2goal_3D: " + str(self.current_arm_distance2goal_3D))               
                reward_step_goal = self.get_reward_step_goal(curr_dist2goal=self.current_arm_distance2goal_3D, prev_dist2goal=self.previous_arm_distance2goal_3D)
            
            else:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] Invalid self.model_mode (reward_step_goal): " + str(self.model_mode) + "!")
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG_INF_2")
                #while 1:
                #    continue

            weighted_reward_step_goal = self.config_mobiman.alpha_step_goal * reward_step_goal # type: ignore

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] reward_step_goal: " + str(reward_step_goal))
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] weighted_reward_step_goal: " + str(weighted_reward_step_goal))

            # Step Reward 3: target
            reward_step_target = 0.0
            if self.flag_action_target:
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] TARGET REWARD IS COMING...")
                
                if self.termination_reason == 'target' and self.model_mode != 1:
                    #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] REACHING TARGET IS NO GOOD MY FRIEND!")
                    reward_step_target = self.get_penalty_step_target()
                
                else:
                    if self.model_mode == 0:
                        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] BASE MOTION")
                        reward_step_target = self.get_reward_step_target()

                    elif self.model_mode == 1:
                        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] ARM MOTION")
                        reward_step_target = self.get_reward_step_com()

                    elif self.model_mode == 2:
                        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] WHOLE-BODY MOTION")
                        reward_step_target = self.get_reward_step_target()

                    else:
                        reward_step_target
                        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] Invalid self.model_mode (reward_step_target): " + str(self.model_mode) + "!")

                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] reward_step_target: " + str(reward_step_target))
            weighted_reward_step_target = self.config_mobiman.alpha_step_target * reward_step_target # type: ignore

            # Total Step Reward
            self.step_reward = round(weighted_reward_step_mode + weighted_reward_step_goal + weighted_reward_step_target, 2)

            ###print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] step_reward: " + str(self.step_reward))
        
        self.episode_reward = round(self.episode_reward + self.step_reward, 2)

        if self.episode_done:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_num: " + str(self.episode_num))

            self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num
            self.total_mean_episode_reward = round(self.total_mean_episode_reward, 2)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] DEBUG INF")
            #while 1:
            #    continue

        self.save_oar_data()

        if self.flag_print_info:
            print("**********************")
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] done: " + str(done))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] step_num: {}".format(self.step_num))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_step_num: {}".format(self.total_step_num))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_num: {}".format(self.episode_num))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] mpc_action_result: {}".format(self.mpc_action_result))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] termination_reason: {}".format(self.termination_reason))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_out_of_boundary: {}".format(self.total_out_of_boundary))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_collisions: {}".format(self.total_collisions))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_rollover: {}".format(self.total_rollover))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_goal: {}".format(self.total_goal))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_max_step: {}".format(self.total_max_step))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_target: {}".format(self.total_target))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_time_horizon: {}".format(self.total_time_horizon))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] step_reward: " + str(self.step_reward))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] episode_reward: {}".format(self.episode_reward))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
            print("**********************")

        self.flag_action_target = False
        self.step_num += 1
        self.total_step_num += 1

        if self.episode_done:
            self.episode_num += 1

            if self.drl_mode == "testing":

                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] TOTAL sample len: " + str(len(self.testing_samples)))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] testing_idx: " + str(self.testing_idx))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] testing_eval_idx: " + str(self.testing_eval_idx))
                #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] CURRENT sample: ")
                #print(self.testing_samples[self.testing_idx])
                
                if self.testing_eval_idx >= self.config_mobiman.n_testing_eval_episodes: 
                    self.testing_eval_idx = 1

                    self.testing_idx += 1
                    if self.testing_idx >= len(self.testing_samples):
                        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] TESTING DONE !!!")
                        self.flag_testing_done = True
                        self.testing_idx -= 1
                else:
                    self.testing_eval_idx += 1

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::compute_reward] END")
        #print("--------------------------------------------------")
        #print("")
        return self.step_reward

    '''
    DESCRIPTION: Load task setup.
    '''
    def load_task_setup(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_task_setup] START")

        self.init_ros_env(ros_node_id=self.ros_node_id, init_flag=self.init_ros_node)

        # domain randomization frequency
        self.texture_randomization_freq = self.config_igibson.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config_igibson.get("object_randomization_freq", None)

        # task
        if "task" not in self.config_igibson:
            self.task = DummyTask(self)
        elif self.config_igibson["task"] == "mobiman_pick":
            if self.flag_print_info:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_task_setup] task: mobiman_pick")            
            ### NUA TODO: SPECIFY NEW TASK ENVIRONMENT!
            self.task = DummyTask(self)
        else:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_task_setup] ERROR: Specified task is invalid!")
            self.task = DummyTask(self)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_task_setup] DEBUG INF")
        #while 1:
        #    continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_task_setup] END")

    '''
    DESCRIPTION: Get observation space of mobiman goal.
    '''
    def get_observation_space_mobiman_goal(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_goal] START")

        obs_goal_x_max = 1.5 * (self.config_mobiman.world_range_x_max - self.config_mobiman.world_range_x_min)
        obs_goal_y_max = 1.5 * (self.config_mobiman.world_range_y_max - self.config_mobiman.world_range_y_min)
        obs_goal_z_max = 1.5 * (self.config_mobiman.world_range_z_max - self.config_mobiman.world_range_z_min)

        ### NUA TODO: GENERALIZE IT FOR TRAJECTORIES OF THE GOAL!
        obs_mobiman_goal_min = np.array([[-obs_goal_x_max,
                                          -obs_goal_y_max,
                                          -obs_goal_z_max]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_mobiman_goal_max = np.array([[obs_goal_x_max,
                                          obs_goal_y_max,
                                          obs_goal_z_max]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_goal] END")

        return obs_mobiman_goal_min, obs_mobiman_goal_max
    
    '''
    DESCRIPTION: Get observation space of mobiman occupancy.
    '''
    def get_observation_space_mobiman_occupancy(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_occupancy] START")

        self.initialize_mobiman_occupancy_obs_config()

        obs_goal_x_max = 1.5 * (self.config_mobiman.world_range_x_max - self.config_mobiman.world_range_x_min)
        obs_goal_y_max = 1.5 * (self.config_mobiman.world_range_y_max - self.config_mobiman.world_range_y_min)
        obs_goal_z_max = 1.5 * (self.config_mobiman.world_range_z_max - self.config_mobiman.world_range_z_min)

        ### NUA TODO: GENERALIZE IT FOR ANY NUMBER OF OBJECTS AND THEIR TRAJECTORIES!
        obs_mobiman_occupancy_min = np.array([[-obs_goal_x_max,
                                               -obs_goal_y_max,
                                               -obs_goal_z_max,
                                               -obs_goal_x_max,
                                               -obs_goal_y_max,
                                               -obs_goal_z_max]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_mobiman_occupancy_max = np.array([[obs_goal_x_max,
                                               obs_goal_y_max,
                                               obs_goal_z_max,
                                               obs_goal_x_max,
                                               obs_goal_x_max,
                                               obs_goal_x_max]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_occupancy] END")

        return obs_mobiman_occupancy_min, obs_mobiman_occupancy_max

    '''
    DESCRIPTION: Get observation space of self collision distance.
    '''
    def get_observation_space_mobiman_selfcoldistance(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_selfcoldistance] START")

        self.initialize_selfcoldistance_config()

        obs_selfcoldistance_min = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_selfcoldistance_max = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_selfcoldistance] END")

        return obs_selfcoldistance_min, obs_selfcoldistance_max
    
    '''
    DESCRIPTION: Get observation space of arm joint position states.
    '''
    def get_observation_space_base_velo(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_base_velo] START")

        obs_base_velo_min = np.array([[self.config_mobiman.obs_base_velo_lat_min,
                                       self.config_mobiman.obs_base_velo_ang_min]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_base_velo_max = np.array([[self.config_mobiman.obs_base_velo_lat_max,
                                       self.config_mobiman.obs_base_velo_ang_max]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_base_velo] END")

        return obs_base_velo_min, obs_base_velo_max

    '''
    DESCRIPTION: Get observation space of arm joint position states.
    '''
    def get_observation_space_mobiman_armstate_pos(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_armstate_pos] START")

        obs_armstate_pos_min = np.full((1, self.config_mobiman.n_armstate), -math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_armstate_pos_max = np.full((1, self.config_mobiman.n_armstate), math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_armstate_pos] END")

        return obs_armstate_pos_min, obs_armstate_pos_max
    
    '''
    DESCRIPTION: Get observation space of arm joint velocity states.
    '''
    def get_observation_space_mobiman_armstate_velo(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_armstate_velo] START")

        obs_armstate_velo_min = np.full((1, self.config_mobiman.n_armstate), self.config_mobiman.obs_joint_velo_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        obs_armstate_velo_max = np.full((1, self.config_mobiman.n_armstate), self.config_mobiman.obs_joint_velo_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_observation_space_mobiman_armstate_velo] END")

        return obs_armstate_velo_min, obs_armstate_velo_max
    
    '''
    DESCRIPTION: Load observation space.
    '''
    def load_observation_space(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] START")

        if self.config_mobiman.observation_space_type == "mobiman_FC":

            obs_mobiman_goal_min, obs_mobiman_goal_max = self.get_observation_space_mobiman_goal()
            obs_mobiman_occupancy_min, obs_mobiman_occupancy_max = self.get_observation_space_mobiman_occupancy()
            obs_selfcoldistance_min, obs_selfcoldistance_max = self.get_observation_space_mobiman_selfcoldistance()
            obs_base_velo_min, obs_base_velo_max = self.get_observation_space_base_velo()
            obs_armstate_pos_min, obs_armstate_pos_max = self.get_observation_space_mobiman_armstate_pos()
            obs_armstate_velo_min, obs_armstate_velo_max = self.get_observation_space_mobiman_armstate_velo()

            obs_space_min = np.concatenate((obs_mobiman_goal_min, 
                                            obs_mobiman_occupancy_min, 
                                            obs_selfcoldistance_min, 
                                            obs_base_velo_min,
                                            obs_armstate_pos_min, 
                                            obs_armstate_velo_min), axis=0)
            obs_space_max = np.concatenate((obs_mobiman_goal_max, 
                                            obs_mobiman_occupancy_max, 
                                            obs_selfcoldistance_max, 
                                            obs_base_velo_max,
                                            obs_armstate_pos_max, 
                                            obs_armstate_velo_max), axis=0)

            #self.obs = obs_space_min
            self.observation_space = gym.spaces.Box(obs_space_min, obs_space_max)

            '''
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_mobiman_goal_min shape: " + str(obs_mobiman_goal_min.shape))
            print(obs_mobiman_goal_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_mobiman_goal_max shape: " + str(obs_mobiman_goal_max.shape))
            print(obs_mobiman_goal_max)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_mobiman_occupancy_min shape: " + str(obs_mobiman_occupancy_min.shape))
            print(obs_mobiman_occupancy_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_mobiman_occupancy_max shape: " + str(obs_mobiman_occupancy_max.shape))
            print(obs_mobiman_occupancy_max)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_selfcoldistance_min shape: " + str(obs_selfcoldistance_min.shape))
            print(obs_selfcoldistance_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_selfcoldistance_max shape: " + str(obs_selfcoldistance_max.shape))
            print(obs_selfcoldistance_max)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_base_velo_min shape: " + str(obs_base_velo_min.shape))
            print(obs_base_velo_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_armstate_pos_min shape: " + str(obs_armstate_pos_min.shape))
            print(obs_armstate_pos_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_armstate_pos_max shape: " + str(obs_armstate_pos_max.shape))
            print(obs_armstate_pos_max)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_armstate_velo_min shape: " + str(obs_armstate_velo_min.shape))
            print(obs_armstate_velo_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_armstate_velo_max shape: " + str(obs_armstate_velo_max.shape))
            print(obs_armstate_velo_max)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_space_min shape: " + str(obs_space_min.shape))
            print(obs_space_min)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_space_max shape: " + str(obs_space_max.shape))
            print(obs_space_max)
            '''

        elif self.config_mobiman.observation_space_type == "mobiman_2DCNN_FC":

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] NEEDS REVIEW: DEBUG INF")
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

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_occgrid_image_min shape: " + str(obs_occgrid_image_min.shape))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_extcoldistancedist_min shape: " + str(obs_extcoldistancedist_min.shape))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] obs_goal_min shape: " + str(obs_goal_min.shape))

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

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] self.observation_space shape: " + str(self.observation_space.shape))
        #print(self.observation_space)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] DEBUG INF")
        #while 1:
        #   continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_observation_space] END")

    '''
    DESCRIPTION: Load action space.
    '''
    def load_action_space(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] START")

        #self.action_space = self.robots[0].action_space

        if self.config_mobiman.action_type == 0:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] DISCRETE!")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] NEEDS REVIEW: DEBUG_INF")
            #while 1:
            #    continue
            
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] n_discrete_action: " + str(self.config_mobiman.n_discrete_action))
            self.action_space = gym.spaces.Discrete(self.config_mobiman.n_discrete_action)
            self.config_mobiman.set_action_shape("Discrete, " + str(self.action_space.n)) # type: ignore

        else:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] CONTINUOUS!")

            action_space_model_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_type_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_min = np.full((1, 3), -1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_ori_min = np.full((1, 3), -math.pi).reshape(self.config_mobiman.fc_obs_shape)
            action_space_min = np.concatenate((action_space_model_min, action_space_target_type_min, action_space_target_pos_min, action_space_target_ori_min), axis=0)

            action_space_model_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_type_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_max = np.full((1, 3), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_ori_max = np.full((1, 3), math.pi).reshape(self.config_mobiman.fc_obs_shape)
            action_space_max = np.concatenate((action_space_model_max, action_space_target_type_max, action_space_target_pos_max, action_space_target_ori_max), axis=0)

            self.action_space = gym.spaces.Box(action_space_min, action_space_max)
            self.config_mobiman.set_action_shape(self.action_space.shape)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space_min shape: " + str(action_space_min.shape))
            #print(action_space_min)

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space_max shape: " + str(action_space_max.shape))
            #print(action_space_max)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_type: " + str(self.config_mobiman.action_type))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space shape: " + str(self.action_space.shape))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space: " + str(self.action_space))
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] DEBUG INF")
        #while 1:
        #    continue
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load_action_space] END")

    '''
    DESCRIPTION: Load environment.
    '''
    def load(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] START")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] START super")
        super(iGibsonEnv, self).load()
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] START load_task_setup")
        self.load_task_setup()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] START load_observation_space")
        self.load_observation_space()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] START load_action_space")
        self.load_action_space()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] DEBUG_INF")   
        #while 1:
        #    continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::load] END")

    '''
    DESCRIPTION: Apply robot's action and return the next state, reward, done and info,
                 following OpenAI Gym's convention
                 
                 :param action: robot actions
                 :return: state: next observation
                 :return: reward: reward of this time step
                 :return: done: whether the episode is terminated
                 :return: info: info dictionary with any useful information
    '''
    def step(self, action):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] START")
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] step_num: " + str(self.step_num))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] total_step_num: " + str(self.total_step_num))

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")   
        #while 1:
        #    continue

        info = {}

        if self.flag_print_info and self.config_mobiman.action_type == 1:
            if action[0] <= 0.3:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BASE MOTION")
            elif action[0] > 0.6:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] ARM MOTION")
            else:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] WHOLE-BODY MOTION")

        # Take action
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BEFORE take_action")
        self.take_action(action)

        # Update data
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BEFORE update_data")
        
        ## NUA NOTE: ALREADY BEING UPDATED IN timer_update!!!
        #self.update_robot_data()
        #self.update_arm_data()
        #self.update_goal_data()
        #self.update_goal_data_wrt_robot()
        #self.update_goal_data_wrt_ee()
        #self.update_target_data()

        # Update observation (state)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BEFORE update_observation")
        self.update_observation()
        state = self.obs

        # Check if episode is done
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BEFORE is_done")
        #done = self.is_done(state)
        done, truncated = self.is_done(state)

        terminated = done
        if truncated:
            terminated = False

        # Compute reward
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] BEFORE compute_reward")
        reward = self.compute_reward(state, done)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")
        #while 1:
        #    continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] END")

        if done and self.automatic_reset:

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] EPISODE DONE")
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] episode_num: " + str(self.episode_num))

            info["last_observation"] = state
            #state = self.reset()

            if self.flag_print_info:
                print("xxxxxxxxxxxxxxxxxxxxxxx END OF EPISODE xxxxxxxxxxxxxxxxxxxxxxx")
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print("")
                print("")

            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::step] DEBUG_INF")
            #while 1:
            #    continue
        else:
            if self.flag_print_info:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF STEP ~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("")
                print("")

        #return state, reward, done, info
        return state, reward, terminated, truncated, info

    '''
    DESCRIPTION: Check whether the given body_id has collision after one simulator step

                 :param body_id: pybullet body id
                 :param ignore_ids: pybullet body ids to ignore collisions with
                 :return: whether the given body_id has collision
    '''
    def check_collision(self, body_id, ignore_ids=[]):
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] DEBUG_INF")
        while 1:
            continue

        self.simulator_step()
        collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]

        if log.isEnabledFor(logging.INFO):  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                log.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) > 0

    '''
    DESCRIPTION: Reset bookkeeping variables for the next new episode.
    '''
    def reset_variables(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset_variables] START")

        self.episode_done = False
        self.reached_goal = False
        self.step_num = 1
        self.current_episode += 1
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_step = 0
        self.collision_links = []

        self.flag_action_target = False
        self.mpc_data_msg = None
        #self.mobiman_goal_obs_msg = None
        #self.mobiman_occupancy_obs_msg = None
        self.selfcoldistance_msg = None
        self.init_update_flag0 = False
        self.init_update_flag1 = False
        self.init_update_flag2 = False
        self.callback_update_flag = False
        self.init_occupancy_data_flag = False

        # Wait for topics
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.mobiman_goal_obs_msg_name) + "...")
        #rospy.wait_for_message(self.ns + self.config_mobiman.mobiman_goal_obs_msg_name, MobimanGoalObservation)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.mobiman_occupancy_obs_msg_name) + "...")
        #rospy.wait_for_message(self.ns + self.config_mobiman.mobiman_occupancy_obs_msg_name, MobimanOccupancyObservation)

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting msg: " + str(self.ns + self.config_mobiman.selfcoldistance_msg_name) + "...")
        rospy.wait_for_message(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info)

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::init_ros_env] Waiting callback_update_flag and init_occupancy_data_flag...")
        while (not self.callback_update_flag) or (not self.init_occupancy_data_flag):
            continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset_variables] DEBUG_INF")
        #while 1:
        #    continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset_variables] END")

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_testing_domain(self):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_testing_domain] START")

        ### NUA TODO: SET ALL PARAMETERS BELOW IN CONFIG!

        ## Robot Pose
        testing_range_robot_pos_x_min = -1.5
        testing_range_robot_pos_x_max = 1.5
        testing_range_robot_n_points_x = 3

        testing_range_robot_pos_y_min = -1.5
        testing_range_robot_pos_y_max = 1.0
        testing_range_robot_n_points_y = 3

        testing_range_robot_yaw_min = 0.0 # -0.5 * math.pi
        testing_range_robot_yaw_max = 1.5 * math.pi # math.pi
        testing_range_robot_n_yaw = 4

        robot_pos_x_lin = np.linspace(testing_range_robot_pos_x_min, testing_range_robot_pos_x_max, testing_range_robot_n_points_x)
        robot_pos_y_lin = np.linspace(testing_range_robot_pos_y_min, testing_range_robot_pos_y_max, testing_range_robot_n_points_y)
        yaw_lin = np.linspace(testing_range_robot_yaw_min, testing_range_robot_yaw_max, testing_range_robot_n_yaw)

        ## Objects
        testing_range_box_pos_x_min = self.config_mobiman.goal_range_min_x
        testing_range_box_pos_x_max = self.config_mobiman.goal_range_max_x
        testing_range_box_n_points_x = 3

        testing_range_box_pos_y_min = self.config_mobiman.goal_range_min_y
        testing_range_box_pos_y_max = self.config_mobiman.goal_range_max_y
        testing_range_box_n_points_y = 1

        testing_range_box_pos_z_min = self.config_mobiman.goal_range_min_z
        testing_range_box_pos_z_max = self.config_mobiman.goal_range_min_z
        testing_range_box_n_points_z = 1

        box_pos_x_lin = np.linspace(testing_range_box_pos_x_min, testing_range_box_pos_x_max, testing_range_box_n_points_x)
        box_pos_y_lin = np.linspace(testing_range_box_pos_y_min, testing_range_box_pos_y_max, testing_range_box_n_points_y)
        box_pos_z_lin = np.linspace(testing_range_box_pos_z_min, testing_range_box_pos_z_max, testing_range_box_n_points_z)

        ### Testing Samples
        ## Format: [robot_pos_x, robot_pos_y, robot_yaw, box_pos_x, box_pos_y, box_pos_z]
        robot_pos_x_mesh, robot_pos_y_mesh, robot_pos_theta, box_pos_x_mesh, box_pos_y_mesh, box_pos_z_mesh = np.meshgrid(robot_pos_x_lin, robot_pos_y_lin, yaw_lin, 
                                                                                                                          box_pos_x_lin, box_pos_y_lin, box_pos_z_lin)
        
        self.testing_samples = np.column_stack((robot_pos_x_mesh.flatten(), robot_pos_y_mesh.flatten(), robot_pos_theta.flatten(), 
                                                box_pos_x_mesh.flatten(), box_pos_y_mesh.flatten(), box_pos_z_mesh.flatten()))
        
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] testing_sample_robot_pose len: " + str(len(self.testing_samples)))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_testing_domain] END")

    '''
    DESCRIPTION: TODO...Gets the initial location of the robot to reset
    '''
    def initialize_robot_pose(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] END")

        init_robot_yaw = 0.0
        if self.drl_mode == "training" and self.config_mobiman.world_name == "conveyor":

            self.config_mobiman.init_robot_pos_range_x_min

            # Set init robot pose
            init_robot_pose_areas_x = []
            init_robot_pose_areas_x.extend(([self.config_mobiman.init_robot_pos_range_x_min, self.config_mobiman.init_robot_pos_range_x_max], 
                                            [self.config_mobiman.init_robot_pos_range_x_min, self.config_mobiman.init_robot_pos_range_x_max], 
                                            [self.config_mobiman.init_robot_pos_range_x_min, self.config_mobiman.init_robot_pos_range_x_max], 
                                            [self.config_mobiman.init_robot_pos_range_x_min, self.config_mobiman.init_robot_pos_range_x_max]))

            init_robot_pose_areas_y = []
            init_robot_pose_areas_y.extend(([self.config_mobiman.init_robot_pos_range_y_min, self.config_mobiman.init_robot_pos_range_y_max], 
                                            [self.config_mobiman.init_robot_pos_range_y_min, self.config_mobiman.init_robot_pos_range_y_max], 
                                            [self.config_mobiman.init_robot_pos_range_y_min, self.config_mobiman.init_robot_pos_range_y_max], 
                                            [self.config_mobiman.init_robot_pos_range_y_min, self.config_mobiman.init_robot_pos_range_y_max]))

            area_idx = random.randint(0, len(init_robot_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.init_robot_pose["x"] = random.uniform(init_robot_pose_areas_x[area_idx][0], init_robot_pose_areas_x[area_idx][1])
            self.init_robot_pose["y"] = random.uniform(init_robot_pose_areas_y[area_idx][0], init_robot_pose_areas_y[area_idx][1])
            self.init_robot_pose["z"] = 0.15
            init_robot_yaw = random.uniform(0.0, 2*math.pi)

            '''
            self.init_robot_pose["x"] = 0.0
            self.init_robot_pose["y"] = 0.0
            self.init_robot_pose["z"] = 0.15
            init_robot_yaw = 0.0
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] DEBUG_WARNING: MANUALLY SET init_robot_pose !!!")
            '''

        elif self.drl_mode == "testing" and self.config_mobiman.world_name == "conveyor":
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] testing_idx: " + str(self.testing_idx))

            self.init_robot_pose["x"] = self.testing_samples[self.testing_idx][0]
            self.init_robot_pose["y"] = self.testing_samples[self.testing_idx][1]
            self.init_robot_pose["z"] = 0.15

            init_robot_yaw = self.testing_samples[self.testing_idx][2]

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] x: " + str(self.init_robot_pose["x"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] y: " + str(self.init_robot_pose["y"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] yaw: " + str(init_robot_yaw))

        robot0_init_quat = Quaternion.from_euler(0, 0, init_robot_yaw)
        self.init_robot_pose["qx"] = robot0_init_quat.x
        self.init_robot_pose["qy"] = robot0_init_quat.y
        self.init_robot_pose["qz"] = robot0_init_quat.z
        self.init_robot_pose["qw"] = robot0_init_quat.w

        self.init_joint_states = {}
        init_arm_joint_pos = [self.cmd_arm_init_j1, self.cmd_arm_init_j2, self.cmd_arm_init_j3, self.cmd_arm_init_j4, self.cmd_arm_init_j5, self.cmd_arm_init_j6]
        init_arm_joint_velo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, jn in enumerate(self.config_mobiman.arm_joint_names):
            self.init_joint_states[jn] = (init_arm_joint_pos[i], init_arm_joint_velo[i])

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] init_joint_states:")
        #print(self.init_joint_states)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] Updated init_robot_pose x: " + str(self.init_robot_pose["x"]) + ", y: " + str(self.init_robot_pose["y"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::initialize_robot_pose] END")

    '''
    DESCRIPTION: TODO...Randomize domain.
                 Object randomization loads new object models with the same poses.
                 Texture randomization loads new materials and textures for the same object models.
    '''
    def randomize_domain(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_domain] START")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_domain] DEBUG_INF")
        #while 1:
        #    continue

        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

        self.initialize_robot_pose()
        self.randomize_env()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::randomize_domain] END")

    '''
    DESCRIPTION: TODO...Reset episode.
    '''
    #def reset(self):
    def reset(self, seed=None, options=None):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] START")
        #print("==================================================")
        #print("==================================================")
        #print("==================================================")
        #print("==================================================")
        #print("==================================================")
        #print("==================================================")
        #print("==================================================")
        
        # We need the following line to seed self.np_random
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE super reset")
        super().reset(seed=seed)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] AFTER super reset")

        info = {}

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] flag_run_sim FALSE")
        self.flag_run_sim = False
        self.mpc_data_received_flag = False
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE randomize_domain")
        self.randomize_domain()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE reset_variables")
        self.reset_variables()
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] AFTER reset_variables")

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] Sleeping 1...")
        #rospy.sleep(0.0)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] DEBUG_INF")
        #while 1:
        #    continue

        init_robot_pos = [self.init_robot_pose["x"], self.init_robot_pose["y"], self.init_robot_pose["z"]]
        init_robot_quat = [self.init_robot_pose["qx"], self.init_robot_pose["qy"], self.init_robot_pose["qz"], self.init_robot_pose["qw"]]
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] init_robot_pos: " + str(init_robot_pos))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] init_robot_quat: " + str(init_robot_quat))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] init_joint_states: " + str(self.init_joint_states))
        
        self.robots[0].set_position_orientation(init_robot_pos, init_robot_quat)
        self.robots[0].set_joint_states(self.init_joint_states)
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] cmd_base: " + str(self.cmd_base))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] cmd_arm: " + str(self.cmd_arm))

        self.cmd_base = self.cmd_base_init
        self.cmd_arm = self.cmd_arm_init
        #cmd = self.cmd_base + self.cmd_arm

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE apply_action")
        #self.robots[0].apply_action(cmd)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE simulator_step")
        #self.simulator_step()

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE simulator.sync")
        #self.simulator.sync(force_sync=True)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] flag_run_sim TRUE")
        self.flag_run_sim = True

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE sleep")
        rospy.sleep(0.5)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] AFTER sleep")

        #if self.init_flag:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] BEFORE get_base_distance2goal_2D")
            #self.previous_base_distance2goal_2D = self.get_base_distance2goal_2D()
            #self.previous_arm_distance2goal_3D = self.get_arm_distance2goal_3D()

        self.update_observation()
        state = self.obs

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] DEBUG_INF")
        #while 1:
        #    continue

        

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] Sleeping 3...")
        #rospy.sleep(5.0)

        #print("####################################")
        #print("####################################")
        #print("####################################")
        #print("####################################")

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reset] END")

        return state, info
    
    '''
    DESCRIPTION: TODO...
    '''
    def render(self):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::render] START")

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::render] END")

    '''
    DESCRIPTION: TODO...
    '''
    def close(self):
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::close] START")

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::close] END")

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
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_quaternion_distance] qdist: " + str(qdist))
        return qdist

    '''
    DESCRIPTION: TODO...Calculate the difference between two angles defined in the range from -pi to pi.
    '''
    def get_angle_difference(self, angle1, angle2, flag_normalize=False):
        diff = angle2 - angle1
        while diff <= -math.pi:
            diff += 2*math.pi
        while diff > math.pi:
            diff -= 2*math.pi

        if flag_normalize:
            diff = abs(diff) / math.pi
        
        return diff

    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.init_robot_pose)
        distance2goal =  round(distance2goal, 2)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.init_robot_pose)
        distance2goal =  round(distance2goal, 2)
        return distance2goal

    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.robot_data)
        distance2goal =  round(distance2goal, 2)
        '''
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
        '''

        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.robot_data)
        distance2goal =  round(distance2goal, 2)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_arm_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.arm_data)
        distance2goal =  round(distance2goal, 2)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_arm_quatdistance2goal(self):
        distance2goal = self.get_quaternion_distance(self.goal_data, self.arm_data)
        distance2goal =  round(distance2goal, 2)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_base_distance2target_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.target_data, self.robot_data)
        distance2goal =  round(distance2goal, 2)
        return distance2goal

    '''
    DESCRIPTION: TODO...Gets the distance to the target
    '''
    def get_arm_distance2target_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.target_data, self.arm_data)
        distance2goal =  round(distance2goal, 2)
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
        distance2goal =  round(distance2goal, 2)

        '''
        if self.model_mode != 0 and (distance2goal > 1 or distance2goal < 0):
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] model_mode: " + str(self.model_mode))
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] distance2goal: " + str(distance2goal))

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_base_yawdistance2target] DEBUG INF")
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
    def get_obs_selfcoldistance(self, check_flag=False):
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] START")

        check_result = True

        selfcoldistance_msg = self.selfcoldistance_msg

        obs_selfcoldistance = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        for i, dist in enumerate(selfcoldistance_msg.distance):
            #csm = selfcoldistance_msg.markers[i*self.config_mobiman.selfcoldistance_n_coeff] # type: ignore
            #p1 = {"x":csm.points[0].x, "y":csm.points[0].y, "z":csm.points[0].z}
            #p2 = {"x":csm.points[1].x, "y":csm.points[1].y, "z":csm.points[1].z} 
            #dist = self.get_euclidean_distance_3D(p1, p2)
            obs_selfcoldistance[i] = dist
            if dist < self.config_mobiman.self_collision_range_min:
                obs_selfcoldistance[i] = self.config_mobiman.self_collision_range_min
            elif dist > self.config_mobiman.self_collision_range_max:
                obs_selfcoldistance[i] = self.config_mobiman.self_collision_range_max
            
            if check_flag:
                if  (obs_selfcoldistance[i] < self.config_mobiman.self_collision_range_min or 
                     obs_selfcoldistance[i] > self.config_mobiman.self_collision_range_max):
                    check_result = False

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] dist " + str(i) + ": " + str(dist))
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] obs_selfcoldistance shape: " + str(obs_selfcoldistance.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] DEBUG INF")
        #while 1:
        #    continue
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_selfcoldistance] END")

        return obs_selfcoldistance, check_result
    
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
    def get_obs_base_velo(self, check_flag=False):
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_base_velo] joint_velo:")
        
        check_result = True

        obs_base_velo_lat = self.robots[0].get_linear_velocity()[0] 
        if obs_base_velo_lat < self.config_mobiman.obs_base_velo_lat_min:
            obs_base_velo_lat = self.config_mobiman.obs_base_velo_lat_min
        elif obs_base_velo_lat > self.config_mobiman.obs_base_velo_lat_max:
            obs_base_velo_lat = self.config_mobiman.obs_base_velo_lat_max

        obs_base_velo_ang = self.robots[0].get_angular_velocity()[2]
        if obs_base_velo_ang < self.config_mobiman.obs_base_velo_ang_min:
            obs_base_velo_ang = self.config_mobiman.obs_base_velo_ang_min
        elif obs_base_velo_ang > self.config_mobiman.obs_base_velo_ang_max:
            obs_base_velo_ang = self.config_mobiman.obs_base_velo_ang_max

        obs_base_velo = np.array([[obs_base_velo_lat,  
                                   obs_base_velo_ang]]).reshape(self.config_mobiman.fc_obs_shape)

        if check_flag:
            if  (obs_base_velo[0] < self.config_mobiman.obs_base_velo_lat_min or 
                 obs_base_velo[0] > self.config_mobiman.obs_base_velo_lat_max or
                 obs_base_velo[1] < self.config_mobiman.obs_base_velo_ang_min or
                 obs_base_velo[1] > self.config_mobiman.obs_base_velo_ang_max):
                check_result = False

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_base_velo] obs_armstate shape: " + str(obs_armstate.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_base_velo] DEBUG_INF")
        #while 1:
        #    continue
        
        return obs_base_velo, check_result

    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_armstate(self, check_flag=False):
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] START")

        check_result = True

        #extcoldistance_arm_msg = self.extcoldistance_arm_msg

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] joint_pos:")
        #print(self.arm_data["joint_pos"])

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] joint_velo:")
        #print(self.arm_data["joint_velo"])
        
        obs_arm_pos = np.full((1, self.config_mobiman.n_armstate), 0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        joint_pos_data = self.arm_data["joint_pos"]
        for i, jp in enumerate(joint_pos_data):
            obs_arm_pos[i] = jp
            #print(str(i) + " -> " + str(jp))

            if check_flag:
                if  (obs_arm_pos[i] < -math.pi or 
                     obs_arm_pos[i] > math.pi):
                    
                    print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] math.pi: " + str(math.pi))

                    check_result = False

        obs_arm_velo = np.full((1, self.config_mobiman.n_armstate), 0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
        joint_velo_data = self.arm_data["joint_velo"]
        for i, jv in enumerate(joint_velo_data):
            obs_arm_velo[i] = jv
            #print(str(i) + " -> " + str(jv))

            if jv < self.config_mobiman.obs_joint_velo_min:
                obs_arm_velo[i] = self.config_mobiman.obs_joint_velo_min

            elif jv > self.config_mobiman.obs_joint_velo_max:
                obs_arm_velo[i] = self.config_mobiman.obs_joint_velo_max

            if check_flag:
                if  (obs_arm_velo[i] < self.config_mobiman.obs_joint_velo_min or 
                     obs_arm_velo[i] > self.config_mobiman.obs_joint_velo_max):
                    
                    print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] obs_arm_velo[i]: " + str(obs_arm_velo[i]))
                    print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] obs_joint_velo_min: " + str(self.config_mobiman.obs_joint_velo_min))
                    print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] obs_joint_velo_max: " + str(self.config_mobiman.obs_joint_velo_max))
                    
                    check_result = False

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] obs_armstate shape: " + str(obs_armstate.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] DEBUG INF")
        #while 1:
        #    continue
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_obs_armstate] START")
            
        return obs_arm_pos, obs_arm_velo, check_result
    
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
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_goal] START")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_goal] DEBUG_INF")
        while 1:
            continue

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
        
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_goal] END")

        return obs_goal

    '''
    DESCRIPTION: Get current observation of mobiman goal.
    '''
    def get_obs_mobiman_goal(self, check_flag=False):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] START")

        check_result = True

        obs_goal_x_max = 1.5 * (self.config_mobiman.world_range_x_max - self.config_mobiman.world_range_x_min)
        obs_goal_y_max = 1.5 * (self.config_mobiman.world_range_y_max - self.config_mobiman.world_range_y_min)
        obs_goal_z_max = 1.5 * (self.config_mobiman.world_range_z_max - self.config_mobiman.world_range_z_min)
        
        obs_goal_x = self.goal_data["x_wrt_robot"]
        if obs_goal_x < -obs_goal_x_max:
            obs_goal_x = -obs_goal_x_max
        elif obs_goal_x > obs_goal_x_max:
            obs_goal_x = obs_goal_x_max

        obs_goal_y = self.goal_data["y_wrt_robot"]
        if obs_goal_y < -obs_goal_y_max:
            obs_goal_y = -obs_goal_y_max
        elif obs_goal_y > obs_goal_y_max:
            obs_goal_y = obs_goal_y_max

        obs_goal_z = self.goal_data["z_wrt_robot"]
        if obs_goal_z < -obs_goal_z_max:
            obs_goal_z = -obs_goal_z_max
        elif obs_goal_z > obs_goal_z_max:
            obs_goal_z = obs_goal_z_max

        obs_mobiman_goal = np.array([[obs_goal_x,
                                      obs_goal_y,
                                      obs_goal_z]]).reshape(self.config_mobiman.fc_obs_shape)

        if check_flag:
            if  (obs_mobiman_goal[0] < -obs_goal_x_max or 
                 obs_mobiman_goal[0] > obs_goal_x_max or
                 obs_mobiman_goal[1] < -obs_goal_y_max or
                 obs_mobiman_goal[1] > obs_goal_y_max or
                 obs_mobiman_goal[2] < -obs_goal_z_max or
                 obs_mobiman_goal[2] > obs_goal_z_max):
                
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_goal_x_max: " + str(obs_goal_x_max))
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_goal_y_max: " + str(obs_goal_y_max))
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_goal_z_max: " + str(obs_goal_z_max))

                check_result = False

        #mobiman_goal_obs_msg = self.mobiman_goal_obs_msg
        #obs_mobiman_goal = np.array(mobiman_goal_obs_msg.obs).reshape(self.config_mobiman.fc_obs_shape)
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] END")

        return obs_mobiman_goal, check_result

    '''
    DESCRIPTION: Get current observation of mobiman occupancy.
    '''
    def get_obs_mobiman_occupancy(self, check_flag=False):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_occupancy] START")

        check_result = True

        obs_occ_x_max = 1.5 * (self.config_mobiman.world_range_x_max - self.config_mobiman.world_range_x_min)
        obs_occ_y_max = 1.5 * (self.config_mobiman.world_range_y_max - self.config_mobiman.world_range_y_min)
        obs_occ_z_max = 1.5 * (self.config_mobiman.world_range_z_max - self.config_mobiman.world_range_z_min)

        obs_occ = []
        for i, occ in enumerate(self.config_mobiman.occupancy_frame_names):
            
            obs_occ_x = self.occupancy_data[occ][0]
            if obs_occ_x < -obs_occ_x_max:
                obs_occ_x = -obs_occ_x_max
            elif obs_occ_x > obs_occ_x_max:
                obs_occ_x = obs_occ_x_max

            obs_occ_y = self.occupancy_data[occ][1]
            if obs_occ_y < -obs_occ_y_max:
                obs_occ_y = -obs_occ_y_max
            elif obs_occ_y > obs_occ_y_max:
                obs_occ_y = obs_occ_y_max

            obs_occ_z = self.occupancy_data[occ][2]
            if obs_occ_z < -obs_occ_z_max:
                obs_occ_z = -obs_occ_z_max
            elif obs_occ_z > obs_occ_z_max:
                obs_occ_z = obs_occ_z_max
            
            obs_occ.append([obs_occ_x, obs_occ_y, obs_occ_z])

            if check_flag:
                if  (obs_occ[i][0] < -obs_occ_x_max or 
                     obs_occ[i][0] > obs_occ_x_max or
                     obs_occ[i][1] < -obs_occ_y_max or
                     obs_occ[i][1] > obs_occ_y_max or
                     obs_occ[i][2] < -obs_occ_z_max or
                     obs_occ[i][2] > obs_occ_z_max):
                    
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_occ_x_max: " + str(obs_occ_x_max))
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_occ_y_max: " + str(obs_occ_y_max))
                    print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_goal] obs_occ_z_max: " + str(obs_occ_z_max))

                    check_result = False

        obs_mobiman_occupancy = np.array([obs_occ]).reshape(self.config_mobiman.fc_obs_shape)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_occupancy] obs_mobiman_occupancy shape: " + str(obs_mobiman_occupancy.shape))
        #print(obs_mobiman_occupancy)

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_occupancy] DEBUG_INF")
        #while 1:
        #    continue

        #mobiman_occupancy_obs_msg = self.mobiman_occupancy_obs_msg
        #obs_mobiman_occupancy = np.array(mobiman_occupancy_obs_msg.obs).reshape(self.config_mobiman.fc_obs_shape)
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_obs_mobiman_occupancy] END")

        return obs_mobiman_occupancy, check_result

    '''
    DESCRIPTION: TODO...
    '''
    def linear_map(self, value, min_val, max_val):
        """
        Maps a value from the range [-1, 1] to a new range specified by the user.
        
        Args:
            value (float): The value to be mapped (within the range [-1, 1]).
            min_val (float): The minimum value of the new range.
            max_val (float): The maximum value of the new range.
        
        Returns:
            float: The mapped value.
        """

        if value <= -1.0:
            return min_val
        elif value >= 1.0:
            return max_val

        mapped_value = 0.5 * (value + 1)
        mapped_value = mapped_value * (max_val - min_val) + min_val

        return mapped_value

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
    def exponential_function(self, x, gamma):
        return np.exp(gamma * x) # type: ignore

    '''
    DESCRIPTION: TODO...
    '''     
    def sigmoid_function(self, x, gamma):
        return (1 / (1 + np.exp(-gamma * x))) # type: ignore

    '''
    DESCRIPTION: TODO... Return the scaled Gaussian with standard deviation sigma.
    '''
    def gaussian_function(self, x, sigma):
        gaussian = np.exp(- (x / sigma)**2)
        scaled_result = 2 * gaussian - 1
        return scaled_result

    '''
    DESCRIPTION: TODO...Calculates the middle point between two points (p1 and p2) in 2D space
                        based on the scaling parameter alpha, and returns its info with respect
                        to the world coordinate frame.

                        Parameters:
                            p1 (list): Coordinates of the first point in [x, y] format.
                            p2 (list): Coordinates of the second point in [x, y] format.
                            alpha (float): Scaling parameter, 0 <= alpha <= 1.

                        Returns:
                            list: Info of the middle point in [x, y, yaw] format.
    '''
    def calculate_middle_point(self, p1, p2, alpha):
        # Calculate the coordinates of the middle point
        x_middle = p1[0] + alpha * (p2[0] - p1[0])
        y_middle = p1[1] + alpha * (p2[1] - p1[1])
        
        # Calculate the orientation of the middle point with respect to the world coordinate frame
        yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        
        return [x_middle, y_middle, yaw]
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_reward_step_goal(self, curr_dist2goal, prev_dist2goal):
        diff_dist2goal = prev_dist2goal - curr_dist2goal
        reward_step_goal = self.config_mobiman.reward_step_goal_scale * diff_dist2goal
        if curr_dist2goal < self.config_mobiman.reward_step_goal_dist_threshold:
            if self.flag_print_info:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_goal] Bro it's so close, but be careful!")
            reward_step_goal += (self.config_mobiman.reward_step_goal_dist_scale * self.config_mobiman.reward_step_goal_scale)
        reward_step_goal = round(reward_step_goal, 2)
        return reward_step_goal

    '''
    DESCRIPTION: TODO...
    '''
    def get_reward_step_target(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_target] START")

        desired_robot_wrt_world_2D = self.calculate_middle_point(self.previous_base_wrt_world_2D[:-1], 
                                                                 self.previous_goal_wrt_world_2D[:-1], 
                                                                 self.config_mobiman.reward_step_target_intermediate_point_scale)
        
        pT = {"x": desired_robot_wrt_world_2D[0], "y": desired_robot_wrt_world_2D[1]}
        pR0 = {"x": self.previous_base_wrt_world_2D[0], "y": self.previous_base_wrt_world_2D[1]}
        pR1 = {"x": self.current_base_wrt_world_2D[0], "y": self.current_base_wrt_world_2D[1]}
        
        '''
        p = Point()
        p.x = desired_robot_wrt_world_2D[0]
        p.y = desired_robot_wrt_world_2D[1]
        p.z = 0.0
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data, frame_name="world")
        '''

        d0 = self.get_euclidean_distance_2D(pT, pR0)
        d1 = self.get_euclidean_distance_2D(pT, pR1)
        
        if (d1 > d0):
            diff_pos_normalized = 1
        else:
            if d0 > 0:
                diff_pos_normalized = d1 / d0
            else:
                diff_pos_normalized = 1
        
        #diff_yaw_normalized = self.get_angle_difference(desired_robot_wrt_world_2D[-1], self.current_base_wrt_world_2D[-1], True)
        #diff_target = 0.5 * (diff_pos_normalized + diff_yaw_normalized)
        
        diff_target = diff_pos_normalized

        reward_step_target = self.config_mobiman.reward_step_target_scale * self.exponential_function(diff_target, self.config_mobiman.reward_step_target_gamma)
        reward_step_target = round(reward_step_target, 2)
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_target] END")

        return reward_step_target
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_penalty_step_target(self):

        pG = {"x": self.previous_goal_wrt_world_2D[0], "y": self.previous_goal_wrt_world_2D[1]}
        pR0 = {"x": self.previous_base_wrt_world_2D[0], "y": self.previous_base_wrt_world_2D[1]}
        pR1 = {"x": self.current_base_wrt_world_2D[0], "y": self.current_base_wrt_world_2D[1]}
        
        d0 = self.get_euclidean_distance_2D(pG, pR0)
        d1 = self.get_euclidean_distance_2D(pG, pR1)
        d2 = self.get_euclidean_distance_2D(pR0, pR1)
        
        d_norm = max([d0,d1,d2])

        diff_pos_normalized = d2 / d_norm

        penalty_step_target = -self.config_mobiman.reward_step_target_scale * self.exponential_function(diff_pos_normalized, self.config_mobiman.reward_step_target_gamma)
        penalty_step_target = round(penalty_step_target, 2)
        return penalty_step_target
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_reward_step_com(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] START")

        if self.mpc_action_result_total_timestep > 0:
            reward_step_com_val = self.mpc_action_result_com_error_norm_total / self.mpc_action_result_total_timestep

            reward_step_com = self.config_mobiman.reward_step_target_com_scale
            if reward_step_com_val > self.config_mobiman.reward_step_target_com_threshold:
                reward_step_com *= -1
        else:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] NO mpc_action_result_total_timestep !!!")
            reward_step_com_val = 0.0

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] mpc_action_result_total_timestep: " + str(self.mpc_action_result_total_timestep))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] mpc_action_result_com_error_norm_total: " + str(self.mpc_action_result_com_error_norm_total))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] reward_step_target_com_threshold: " + str(self.config_mobiman.reward_step_target_com_threshold))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] reward_step_com_val: " + str(reward_step_com_val))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] reward_step_com: " + str(reward_step_com))

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::get_reward_step_com] END")

        return reward_step_com

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_target2goal_diff_func(self, curr_target2goal, prev_target2goal):

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reward_step_target2goal_diff_func] DEPRECATED: DEBUG_INF")
        while 1:
            continue

        diff_target2goal = prev_target2goal - curr_target2goal
        reward_step_target2goal = self.config_mobiman.reward_step_target2goal * self.gaussian_function(diff_target2goal-self.config_mobiman.reward_step_goal_mu, self.config_mobiman.reward_step_goal_sigma)
        return reward_step_target2goal

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_target2goal_curr_func(self, curr_target2goal):

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reward_step_target2goal_curr_func] DEPRECATED: DEBUG_INF")
        while 1:
            continue

        reward_step_target2goal = self.config_mobiman.reward_step_target2goal * self.gaussian_function(curr_target2goal-self.config_mobiman.reward_step_target2goal_mu_last_step, self.config_mobiman.reward_step_target2goal_sigma_last_step)
        return reward_step_target2goal
    
    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_target2goal_func(self, curr_target2goal, prev_target2goal):

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reward_step_target2goal_func] DEPRECATED: DEBUG_INF")
        while 1:
            continue

        distance2goal = self.get_base_distance2goal_2D()
        if distance2goal < self.config_mobiman.last_step_distance_threshold: # type: ignore
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reward_step_target2goal_func] WITHIN LAST STEP DISTANCE!")
            return self.reward_step_target2goal_curr_func(curr_target2goal)
        else:
            return self.reward_step_target2goal_diff_func(curr_target2goal, prev_target2goal)

    '''
    DESCRIPTION: TODO...
    '''
    def reward_step_time_horizon_func(self, dt_action):

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::reward_step_time_horizon_func] DEPRECATED: DEBUG_INF")
        while 1:
            continue

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
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] START")
        
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] DEBUG_INF")
        while 1:
            continue

        selfcoldistance = self.get_obs_selfcoldistance() 
        extcoldistance_base = self.get_obs_extcoldistance_base()
        extcoldistance_arm = self.get_obs_extcoldistance_arm()
        pointsonrobot = self.get_pointsonrobot()
        
        for dist in selfcoldistance:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] selfcoldistance dist: " + str(dist))
            if dist < self.config_mobiman.self_collision_range_min:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] SELF COLLISION")
                self.episode_done = True
                self.termination_reason = 'collision'
                return True
            
        for dist in extcoldistance_base:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] extcoldistance_base dist: " + str(dist))
            if dist < self.config_mobiman.ext_collision_range_base_min:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] EXT BASE COLLISION")
                self.episode_done = True
                self.termination_reason = 'collision'
                return True
            
        for dist in extcoldistance_arm:
            #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] extcoldistance_arm dist: " + str(dist))
            if dist < self.config_mobiman.ext_collision_range_arm_min:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] EXT ARM COLLISION")
                self.episode_done = True
                self.termination_reason = 'collision'
                return True

        for por in pointsonrobot:
            if por.z < self.config_mobiman.ext_collision_range_base_min:
                print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] GROUND COLLISION ")
                self.episode_done = True
                self.termination_reason = 'collision'
                return True

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_collision] END")

        return False
    
    '''
    DESCRIPTION: TODO...
    value.
    '''
    def check_rollover(self):
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] START")
        
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] DEBUG_INF")
        while 1:
            continue

        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] pitch: " + str(self.robot_data["pitch"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] rollover_pitch_threshold: " + str(self.config_mobiman.rollover_pitch_threshold))
        # Check pitch
        if self.robot_data["pitch"] > self.config_mobiman.rollover_pitch_threshold:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] PITCH ROLLOVER!!!")
            self.episode_done = True
            self.termination_reason = 'rollover'
            return True
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] roll: " + str(self.robot_data["roll"]))
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] rollover_roll_threshold: " + str(self.config_mobiman.rollover_roll_threshold))
        # Check roll
        if self.robot_data["roll"] > self.config_mobiman.rollover_roll_threshold:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] ROLL ROLLOVER!!!")
            self.episode_done = True
            self.termination_reason = 'rollover'
            return True
        
        #print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::check_rollover] END")

        return False
    
    def transform_pose(self, input_pose, target_frame):
        listener = tf.TransformListener()  # Create a TF listener

        # Wait for the transform to be available
        listener.waitForTransform(target_frame, input_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))

        try:
            # Transform the pose to the target frame
            transformed_pose = listener.transformPose(target_frame, input_pose)
            return transformed_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to transform pose")
            return None

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
        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::save_oar_data] START")

        obs_data = self.obs.reshape((-1)) # type: ignore

        # Save Observation-Action-Reward Data
        if self.episode_num == 1 and self.step_num == 1:
            self.oars_data['log_file'].append(self.config_mobiman.log_file)
        else:
            self.oars_data['log_file'].append("")
        self.oars_data['episode_index'].append(self.episode_num)
        self.oars_data['step_index'].append(self.step_num)
        self.oars_data['observation'].append(obs_data.tolist())
        self.oars_data['action'].append(self.step_action)
        self.oars_data['reward'].append(self.step_reward)
        self.oars_data['result'].append(self.termination_reason)
        
        if self.drl_mode == "testing":
            self.oars_data['testing_index'].append(self.testing_idx)
            self.oars_data['testing_state'].append(self.testing_samples[self.testing_idx])
            self.oars_data['testing_eval_index'].append(self.testing_eval_idx)

            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::save_oar_data] testing result: " + str(self.termination_reason))

        if  self.episode_done:
            self.oars_data['log_file'].append("")
            self.oars_data['episode_index'].append(None)
            self.oars_data['step_index'].append(None)
            self.oars_data['observation'].append([])
            self.oars_data['action'].append([])
            self.oars_data['reward'].append(None)
            self.oars_data['result'].append([])

            if self.drl_mode == "testing":
                self.oars_data['testing_index'].append(None)
                self.oars_data['testing_state'].append([])
                self.oars_data['testing_eval_index'].append(None)

        self.data = pd.DataFrame(self.oars_data)
        self.data.to_csv(self.oar_data_file)
        del self.data
        gc.collect()

        if self.flag_print_info:
            print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::save_oar_data] END")

    '''
    DESCRIPTION: TODO...Save a sequence of Trajectories.

        Args:
            path: Trajectories are saved to this path.
            trajectories: The trajectories to save.
    '''
    def write_oar_data(self) -> None:
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::write_oar_data] START")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::write_oar_data] DEBUG_INF")
        while 1:
            continue

        path = self.config_mobiman.data_folder_path + self.ns[1:-1] + "_oar_data.pkl"
        trajectories = self.oar_data
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump(trajectories, f)

        # Ensure atomic write
        os.replace(tmp_path, path)

        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::write_oar_data] Written Observation-Action-Reward data!")
        print("[" + self.ns + "][igibson_env_jackalJaco::iGibsonEnv::write_oar_data] END")

