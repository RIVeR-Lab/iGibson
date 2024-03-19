#!/usr/bin/python3

'''
LAST UPDATE: 2024.02.21

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

import igibson
from igibson.utils.utils import parse_config

### NUA NOTE: USING THE CUSTOMIZED VERSION FOR MOBIMAN!
#from igibson.envs.igibson_env import iGibsonEnv
from igibson.envs.igibson_env_jackalJaco import iGibsonEnv

from drl.mobiman_drl_config import * # type: ignore 
from drl.mobiman_drl_custom_policy import * # type: ignore

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO, SAC, DDPG, A2C, DQN
    #from stable_baselines3.common.evaluation import evaluate_policy
    #from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback

except ModuleNotFoundError:
    print("[drl_training_sb3_mobiman_jackalJaco] stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)

'''
DESCRIPTION: TODO...
'''
def createFileName():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")
    return timestampStr

'''
DESCRIPTION: TODO...
'''
def read_data(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array(next(reader))
        for row in reader:
            data_row = np.array(row)
            data = np.vstack((data, data_row))
        return data
        
'''
DESCRIPTION: TODO...
'''
def write_data(file, data):
    file_status = open(file, 'a')
    with file_status:
        write = csv.writer(file_status)
        write.writerows(data)
        print("[drl_training_sb3_mobiman_jackalJaco::write_data] Data is written in " + str(file))

'''
DESCRIPTION: TODO...
'''
def read_data_size(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array(next(reader))
        i = 0
        for row in reader:
            i += 1
        return i

'''
DESCRIPTION: TODO...
'''
def get_param_value_from_training_log(log_path, param_name):

    with open(log_path) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == param_name:
                return row[1]

'''
DESCRIPTION: TODO...
'''
def print_array(arr):
    for i in range(len(arr)):
        print(str(i) + " -> " + str(arr[i]))

'''
DESCRIPTION: TODO...
'''
def print_training_log(log_path):

    with open(log_path) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            print("[drl_training_sb3_mobiman_jackalJaco::print_training_log] Line " + str(line_count) + " -> " + str(row[0]) + ": " + str(row[1]))
            line_count += 1

'''
DESCRIPTION: TODO...
'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict): # type: ignore
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

def main(selection="user", headless=False, short_exec=False):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80) # type: ignore

    rospack = rospkg.RosPack()
    mobiman_path = rospack.get_path('mobiman_simulation') + "/"

    ## Initialize the parameters
    flag_print_info = rospy.get_param('flag_print_info', False)
    training_log_name = rospy.get_param('training_log_name', "")
    igibson_config_file = rospy.get_param('igibson_config_file', "")
    flag_drl = rospy.get_param('flag_drl', True)
    #mode = rospy.get_param('mode', "")
    rl_algorithm = rospy.get_param('rl_algorithm', "")
    motion_planning_algorithm = rospy.get_param('motion_planning_algorithm', "")
    observation_space_type = rospy.get_param('observation_space_type', "")
    world_name = rospy.get_param('world_name', "")
    task_and_robot_environment_name = rospy.get_param('task_and_robot_environment_name', "")
    n_robot = rospy.get_param('n_robot', 0)
    data_path = rospy.get_param('drl_data_path', "")
    learning_rate = rospy.get_param('learning_rate', 0)
    n_steps = rospy.get_param('n_steps', 0)
    batch_size = rospy.get_param('batch_size', 0)
    ent_coef = rospy.get_param('ent_coef', 0)
    training_timesteps = rospy.get_param('training_timesteps', 0)
    max_episode_steps = rospy.get_param('max_episode_steps', 0)
    initial_training_path = rospy.get_param('initial_training_path', "")
    training_checkpoint_freq = rospy.get_param('training_checkpoint_freq', 0)
    plot_title = rospy.get_param('plot_title', "")
    plot_moving_average_window_size_timesteps = rospy.get_param('plot_moving_average_window_size_timesteps', 0)
    plot_moving_average_window_size_episodes = rospy.get_param('plot_moving_average_window_size_episodes', 0)

    print("[drl_training_sb3_mobiman_jackalJaco::main] flag_print_info: " + str(flag_print_info))
    print("[drl_training_sb3_mobiman_jackalJaco::main] training_log_name: " + str(training_log_name))
    print("[drl_training_sb3_mobiman_jackalJaco::main] igibson_config_file: " + str(igibson_config_file))
    print("[drl_training_sb3_mobiman_jackalJaco::main] flag_drl: " + str(flag_drl))
    #print("[drl_training_sb3_mobiman_jackalJaco::main] mode: " + str(mode))
    print("[drl_training_sb3_mobiman_jackalJaco::main] rl_algorithm: " + str(rl_algorithm))
    print("[drl_training_sb3_mobiman_jackalJaco::main] motion_planning_algorithm: " + str(motion_planning_algorithm))
    print("[drl_training_sb3_mobiman_jackalJaco::main] observation_space_type: " + str(observation_space_type))
    print("[drl_training_sb3_mobiman_jackalJaco::main] world_name: " + str(world_name))
    print("[drl_training_sb3_mobiman_jackalJaco::main] task_and_robot_environment_name: " + str(task_and_robot_environment_name))
    print("[drl_training_sb3_mobiman_jackalJaco::main] n_robot: " + str(n_robot))
    print("[drl_training_sb3_mobiman_jackalJaco::main] data_path: " + str(data_path))
    print("[drl_training_sb3_mobiman_jackalJaco::main] learning_rate: " + str(learning_rate))
    print("[drl_training_sb3_mobiman_jackalJaco::main] n_steps: " + str(n_steps))
    print("[drl_training_sb3_mobiman_jackalJaco::main] batch_size: " + str(batch_size))
    print("[drl_training_sb3_mobiman_jackalJaco::main] ent_coef: " + str(ent_coef))
    print("[drl_training_sb3_mobiman_jackalJaco::main] training_timesteps: " + str(training_timesteps))
    print("[drl_training_sb3_mobiman_jackalJaco::main] max_episode_steps: " + str(max_episode_steps))
    print("[drl_training_sb3_mobiman_jackalJaco::main] initial_training_path: " + str(initial_training_path))
    print("[drl_training_sb3_mobiman_jackalJaco::main] training_checkpoint_freq: " + str(training_checkpoint_freq))
    print("[drl_training_sb3_mobiman_jackalJaco::main] plot_title: " + str(plot_title))
    print("[drl_training_sb3_mobiman_jackalJaco::main] plot_moving_average_window_size_timesteps: " + str(plot_moving_average_window_size_timesteps))
    print("[drl_training_sb3_mobiman_jackalJaco::main] plot_moving_average_window_size_episodes: " + str(plot_moving_average_window_size_episodes))

    ## Create the folder name that the data is kept
    data_file_tag = createFileName()
    data_folder_tag = data_file_tag + "_" + rl_algorithm + "_mobiman" # type: ignore
    data_name = data_folder_tag + "/" # type: ignore
    data_folder_path = mobiman_path + data_path + data_name

    os.makedirs(data_folder_path, exist_ok=True)

    new_trained_model_file = data_folder_path + "trained_model"
    training_log_file = data_folder_path + training_log_name + ".csv"
    tensorboard_log_path = data_folder_path + rl_algorithm + "_tensorboard/"

    ## Keep all parameters in an array to save
    training_log_data = []
    training_log_data.append(["igibson_config_file", igibson_config_file])
    training_log_data.append(["rl_algorithm", rl_algorithm])
    training_log_data.append(["motion_planning_algorithm", motion_planning_algorithm])
    training_log_data.append(["observation_space_type", observation_space_type])
    training_log_data.append(["world_name", world_name])
    training_log_data.append(["task_and_robot_environment_name", task_and_robot_environment_name])
    training_log_data.append(["n_robot", n_robot])
    training_log_data.append(["data_path", data_path])
    training_log_data.append(["learning_rate", learning_rate])
    training_log_data.append(["n_steps", n_steps])
    training_log_data.append(["batch_size", batch_size])
    training_log_data.append(["ent_coef", ent_coef])
    training_log_data.append(["training_timesteps", training_timesteps])
    training_log_data.append(["max_episode_steps", max_episode_steps])
    training_log_data.append(["initial_training_path", initial_training_path])
    training_log_data.append(["training_checkpoint_freq", training_checkpoint_freq])
    training_log_data.append(["plot_title", plot_title])
    training_log_data.append(["plot_moving_average_window_size_timesteps", plot_moving_average_window_size_timesteps])
    training_log_data.append(["plot_moving_average_window_size_episodes", plot_moving_average_window_size_episodes])

    ## Write all parameters into the log file of the training
    write_data(training_log_file, training_log_data)

    #print("[drl_training_sb3_mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue
    
    ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
    igibson_config_path = igibson.ros_path + "/config/" + igibson_config_file + ".yaml" # type: ignore
    igibson_config_data = yaml.load(open(igibson_config_path, "r"), Loader=yaml.FullLoader)

    igibson_config = parse_config(igibson_config_data)

    pybullet_mode = igibson_config["mode"]
    action_timestep = igibson_config["action_timestep"]
    physics_timestep = igibson_config["physics_timestep"]
    use_pb_gui = igibson_config["use_pb_gui"]
    objects = igibson_config["objects"]
    tensorboard_log_dir = igibson.ros_path + "/log"
    num_environments = n_robot

    print("[drl_training_sb3_mobiman_jackalJaco::main] ros_path: " + str(igibson.ros_path))
    print("[drl_training_sb3_mobiman_jackalJaco::main] config_file: " + str(igibson_config_file))
    print("[drl_training_sb3_mobiman_jackalJaco::main] config_path: " + str(igibson_config_path))
    print("[drl_training_sb3_mobiman_jackalJaco::main] config_data: " + str(igibson_config_data))

    print("[drl_training_sb3_mobiman_jackalJaco::main] flag_drl: " + str(flag_drl))
    print("[drl_training_sb3_mobiman_jackalJaco::main] pybullet_mode: " + str(pybullet_mode))
    print("[drl_training_sb3_mobiman_jackalJaco::main] action_timestep: " + str(action_timestep))
    print("[drl_training_sb3_mobiman_jackalJaco::main] physics_timestep: " + str(physics_timestep))
    print("[drl_training_sb3_mobiman_jackalJaco::main] use_pb_gui: " + str(use_pb_gui))
    
    print("[drl_training_sb3_mobiman_jackalJaco::main] tensorboard_log_dir: " + str(tensorboard_log_dir))

    #print("[drl_training_sb3_mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    # Create environments
    print("[drl_training_sb3_mobiman_jackalJaco::main] BEFORE make_env")
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=igibson_config_path,
                mode=pybullet_mode,
                drl_mode="training",
                action_timestep=action_timestep,
                physics_timestep=physics_timestep,
                init_ros_node=True,
                ros_node_id=rank,
                use_pb_gui=use_pb_gui,
                automatic_reset=True,
                data_folder_path=data_folder_path,
                log_file=training_log_file,
                objects=objects,
                flag_drl=flag_drl,
                flag_print_info=flag_print_info           
            )
            env.reset(seed + rank) # type: ignore
            return env

        set_random_seed(seed)
        return _init
    print("[drl_training_sb3_mobiman_jackalJaco::main] AFTER make_env")

    # Set multi-process
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    env = VecMonitor(env)

    #print("[drl_training_sb3_mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    # Set learning model
    if observation_space_type == "mobiman_FC":
        
        n_actions = env.action_space
        print("[drl_training_sb3_mobiman_jackalJaco::__main__] n_actions: " + str(n_actions))
        
        if rl_algorithm == "SAC":
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] SAC IS IN CHARGE!")
            policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]), 
                                 activation_fn=th.nn.ReLU)
            
            model = SAC(
                "MlpPolicy", 
                env, 
                learning_rate=learning_rate, # type: ignore
                learning_starts=500,
                buffer_size=5000,
                batch_size=batch_size, # type: ignore
                train_freq=4,
                ent_coef='auto', # type: ignore
                tensorboard_log=tensorboard_log_path, 
                policy_kwargs=policy_kwargs, 
                device="cuda", 
                verbose=1)
        
        elif rl_algorithm == "DDPG":
            #print("[drl_training_sb3_mobiman_jackalJaco::__main__] NUA TODO: DDPG IS NOT IMPLEMENTED YET!")
            #print("[drl_training_sb3_mobiman_jackalJaco::__main__] DEBUG_INF")
            #while 1:
            #    continue

            print("[drl_training_sb3_mobiman_jackalJaco::__main__] DDPG IS IN CHARGE!")
            policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]), 
                                 activation_fn=th.nn.ReLU)
            
            model = DDPG(
                "MlpPolicy", 
                env, 
                learning_rate=learning_rate, # type: ignore
                learning_starts=500,
                batch_size=batch_size, # type: ignore
                train_freq=10,
                #ent_coef='auto', # type: ignore
                tensorboard_log=tensorboard_log_path, 
                policy_kwargs=policy_kwargs, 
                device="cuda", 
                verbose=1)

        elif rl_algorithm == "A2C":
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] A2C IS NOT IMPLEMENTED YET!")
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] DEBUG_INF")
            while 1:
                continue

        elif rl_algorithm == "DQN":
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] DQN IS IN CHARGE!")
            policy_kwargs = dict(net_arch=[400, 300], 
                                 activation_fn=th.nn.ReLU)
            
            model = DQN(
                "MlpPolicy", 
                env, 
                learning_rate=learning_rate, # type: ignore
                buffer_size=5000,
                learning_starts=500,
                batch_size=250, # type: ignore
                train_freq=5,
                target_update_interval=500,
                #ent_coef='auto', # type: ignore
                tensorboard_log=tensorboard_log_path, 
                policy_kwargs=policy_kwargs, 
                device="cuda", 
                verbose=1)
            
        else:
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] PPO IS IN CHARGE!")
            policy_kwargs = dict(net_arch=dict(pi=[400, 300], vf=[400, 300]), 
                                 activation_fn=th.nn.ReLU)
            
            model = PPO(
                "MlpPolicy", 
                env, 
                learning_rate=learning_rate, # type: ignore
                n_steps=n_steps, # type: ignore
                batch_size=batch_size, # type: ignore
                ent_coef=ent_coef, # type: ignore
                tensorboard_log=tensorboard_log_path, 
                policy_kwargs=policy_kwargs, 
                device="cuda", 
                verbose=1)
        
        if initial_training_path == "":
            total_training_timesteps = training_timesteps
            print("[drl_training_sb3_mobiman_jackalJaco::__main__] No initial_trained_model is loaded!")
            rospy.logdebug("[drl_training_sb3_mobiman_jackalJaco::__main__] No initial_trained_model is loaded!")

        else:
            initial_trained_model_path = mobiman_path + data_path + initial_training_path + "trained_model" # type: ignore
            initial_trained_model = initial_trained_model_path
            
            if rl_algorithm == "SAC":
                model = SAC.load(initial_trained_model, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
            
            elif rl_algorithm == "DDPG":
                model = DDPG.load(initial_trained_model, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
            
            elif rl_algorithm == "A2C":
                model = A2C.load(initial_trained_model, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
            
            else:
                model = PPO.load(initial_trained_model, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
            #model.set_env(env)

            training_log_path = mobiman_path + data_path + initial_training_path + training_log_name + ".csv"
            total_training_timesteps = int(get_param_value_from_training_log(training_log_path, "total_training_timesteps")) + training_timesteps # type: ignore
            print("[mobiman_drl_training::__main__] Loaded initial_trained_model: " + initial_trained_model)
            #rospy.logdebug("[mobiman_drl_training::__main__] Loaded initial_trained_model: " + initial_trained_model)

        '''
        # Obtain the arguments/parameters for the policy and create the PPO model
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
        )
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        model = PPO("MultiInputPolicy", env, n_steps=512, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
        print(model.policy)
        '''

    elif observation_space_type == "mobiman_2DCNN_FC":
    
        print("[mobiman_drl_training::__main__] observation_space_type: " + str(observation_space_type))
        n_actions = env.action_space
        #n_actions = env.action_space.shape[-1]
        print("[mobiman_drl_training::__main__] n_actions: " + str(n_actions))

        policy_kwargs = dict(features_extractor_class=mobiman_2DCNN_FC_Policy, net_arch=[dict(pi=[600, 400], vf=[600, 400])],) # type: ignore
        model = PPO(
            "MultiInputPolicy", 
            env, 
            learning_rate=learning_rate, # type: ignore
            n_steps=n_steps, # type: ignore
            batch_size=batch_size, # type: ignore
            ent_coef=ent_coef, # type: ignore
            tensorboard_log=tensorboard_log_path, 
            policy_kwargs=policy_kwargs, 
            device="cuda", 
            verbose=1)


    '''
    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=os.path.join(igibson.configs_path, config_file),
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )
    '''

    '''
    print("BEFORE evaluate_policy 0")
    # Random Agent, evaluation before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 0")
    '''

    checkpoint_callback = CheckpointCallback(save_freq=training_checkpoint_freq, save_path=data_folder_path + '/training_checkpoints/', name_prefix='trained_model') # type: ignore

    print("[drl_training_sb3_mobiman_jackalJaco::main] BEFORE learn")
    start_learning = time.time()

    print("[drl_training_sb3_mobiman_jackalJaco::main] training_timesteps: " + str(training_timesteps))

    model.learn(training_timesteps, callback=checkpoint_callback, progress_bar=True) # type: ignore
    end_learning = time.time()
    print("[drl_training_sb3_mobiman_jackalJaco::main] AFTER learn")

    model.save(new_trained_model_file) # type: ignore

    learning_time = (end_learning - start_learning) / 60

    #total_training_episodes = read_data_size(data_folder_path + "training_data.csv")

    training_log_data = []
    #training_log_data.append(["total_training_episodes", total_training_episodes])
    training_log_data.append(["total_training_timesteps", total_training_timesteps])
    training_log_data.append(["learning_time[min]", learning_time])

    print("--------------")
    print("[mobiman_drl_training::__main__] End of training!")
    print("[mobiman_drl_training::__main__] learning_time[min]: " + str(learning_time))
    print("--------------")
    #rospy.logdebug("[mobiman_drl_training::__main__] End of training!")

    ## Write all results into the log file of the training
    write_data(training_log_file, training_log_data)

    '''
    print("BEFORE evaluate_policy 1")
    # Evaluate the policy after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 1")

    # Save the trained model and delete it
    model.save("ckpt")
    del model

    # Reload the trained model from file
    model = PPO.load("ckpt")

    print("BEFORE evaluate_policy 2")
    # Evaluate the trained model loaded from file
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 2")
    '''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
