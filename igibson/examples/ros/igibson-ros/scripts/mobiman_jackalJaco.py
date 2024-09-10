#!/usr/bin/python3

'''
LAST UPDATE: 2024.01.15

AUTHOR: Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:

NUA TODO:
- 
'''

import rospy
import rospkg
import logging
import os
import csv
from datetime import datetime
import time
from typing import Callable
import yaml
import numpy as np

import igibson # type: ignore
from igibson.utils.utils import parse_config # type: ignore

### NUA NOTE: USING THE CUSTOMIZED VERSION FOR MOBIMAN!
#from igibson.envs.igibson_env import iGibsonEnv
from igibson.envs.igibson_env_jackalJaco import iGibsonEnv # type: ignore

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)

def main():
    """
    Example mobiman+igibson
    - Loads a scene
    - Starts the simulation of the robot
    - Publishes ROS topics 
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80) # type: ignore

    print("[mobiman_jackalJaco::main] START")

    rospack = rospkg.RosPack()
    mobiman_path = rospack.get_path('mobiman_simulation') + "/"

    ## Initialize the parameters
    flag_print_info = rospy.get_param('flag_print_info', False)
    igibson_config_file = rospy.get_param('igibson_config_file', "")
    mobiman_mode = rospy.get_param('mode', "")

    print("[mobiman_jackalJaco::main] igibson_config_file: " + str(igibson_config_file))
    print("[mobiman_jackalJaco::main] mobiman_mode: " + str(mobiman_mode))

    #print("[mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue
    
    ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
    igibson_config_path = igibson.ros_path + "/config/" + igibson_config_file + ".yaml" # type: ignore
    igibson_config_data = yaml.load(open(igibson_config_path, "r"), Loader=yaml.FullLoader)

    igibson_config = parse_config(igibson_config_data)

    n_robot = rospy.get_param('n_robot', 0)
    flag_drl = rospy.get_param('flag_drl', False)
    mode = igibson_config["mode"]
    action_timestep = igibson_config["action_timestep"]
    physics_timestep = igibson_config["physics_timestep"]
    #render_timestep = igibson_config["render_timestep"]
    use_pb_gui = igibson_config["use_pb_gui"]
    objects = igibson_config["objects"]
    tensorboard_log_dir = igibson.ros_path + "/log"
    num_environments = n_robot+1

    print("[mobiman_jackalJaco::main] flag_print_info: " + str(flag_print_info))
    print("[mobiman_jackalJaco::main] ros_path: " + str(igibson.ros_path))
    print("[mobiman_jackalJaco::main] config_file: " + str(igibson_config_file))
    print("[mobiman_jackalJaco::main] config_path: " + str(igibson_config_path))
    print("[mobiman_jackalJaco::main] config_data: " + str(igibson_config_data))
    
    print("[mobiman_jackalJaco::main] n_robot: " + str(n_robot))
    print("[mobiman_jackalJaco::main] flag_drl: " + str(flag_drl))
    print("[mobiman_jackalJaco::main] mode: " + str(mode))
    print("[mobiman_jackalJaco::main] action_timestep: " + str(action_timestep))
    print("[mobiman_jackalJaco::main] physics_timestep: " + str(physics_timestep))
    #print("[mobiman_jackalJaco::main] render_timestep: " + str(render_timestep))
    print("[mobiman_jackalJaco::main] use_pb_gui: " + str(use_pb_gui))
    
    print("[mobiman_jackalJaco::main] tensorboard_log_dir: " + str(tensorboard_log_dir))

    #print("[mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    # Create environments
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=igibson_config_path,
                mode=mode,
                action_timestep=action_timestep,
                physics_timestep=physics_timestep,
                init_ros_node=True,
                ros_node_id=rank,
                scene_id=rank,
                use_pb_gui=use_pb_gui,
                automatic_reset=True,
                objects=objects,
                flag_drl=flag_drl,   
                flag_print_info=flag_print_info      
            )
            env.seed(seed + rank) # type: ignore
            return env

        set_random_seed(seed)
        return _init

    # Set multi-process
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)]) # type: ignore
    env = VecMonitor(env)

    print("[mobiman_jackalJaco::main] DEBUG_INF")
    while 1:
        continue

    print("[mobiman_jackalJaco::main] END")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()