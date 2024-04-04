#!/usr/bin/python3

'''
LAST UPDATE: 2024.03.20

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
import matplotlib.pyplot as plt

from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

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
    from stable_baselines3.common.evaluation import evaluate_policy
    #from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    #from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.monitor import Monitor
    #from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback

except ModuleNotFoundError:
    print("[drl_testing_sb3_mobiman_jackalJaco] stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
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
def write_data(filepath, filename, data):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file_status = open(filepath + filename, 'a')
    with file_status:
        write = csv.writer(file_status)
        write.writerows(data)
        print("[drl_testing_sb3_mobiman_jackalJaco::write_data] Data is written in " + str(filepath+filename))

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
def get_param_value_from_testing_log(log_path, param_name):

    with open(log_path + 'testing_log.csv') as csv_file:
        
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
def print_testing_log(log_path):

    with open(log_path + 'testing_log.csv') as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            print("[drl_testing_sb3_mobiman_jackalJaco::print_testing_log] Line " + str(line_count) + " -> " + str(row[0]) + ": " + str(row[1]))
            line_count += 1

'''
DESCRIPTION: TODO...
'''
def get_successful_avg(data, success_data):
    total_val = 0.0
    counter = 0

    for d in range(len(success_data)):
        if success_data[d] > 0:
            total_val += data[d]
        counter += 1
    return total_val / counter

'''
DESCRIPTION: TODO...
'''
def get_color_array(success_data):
    color_array = []
    for d in success_data:
        if d == 0.0:
            color_array.append("red")
        else:
            color_array.append("blue")
    return color_array

'''
DESCRIPTION: TODO... plot the results
    :param log_path: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
'''
def plot_testing_result(log_path):

    success_data = []
    duration_data = []
    path_length_data = []

    with open(log_path + 'testing_result_log.csv') as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count > 0:
                success_data.append(float(row[0]))
                duration_data.append(float(row[1]))
                path_length_data.append(float(row[2]))
            line_count += 1

    label_x = list(map(str, range(1,len(duration_data)+1,1)))
    color_array = get_color_array(success_data)
    
    avg_success = sum(success_data) / len(success_data)
    plt.figure(1)
    plt.bar(label_x, success_data, color=color_array)
    plt.axhline(avg_success, color='blue', linewidth=2)
    plt.xlabel('Episode Index')
    plt.ylabel('Navigation Success')
    plt.title("Navigation Success")
    plt.savefig(log_path + '/nav_success.png')

    avg_duration = get_successful_avg(duration_data, success_data)
    plt.figure(2)
    plt.bar(label_x, duration_data, color=color_array)
    plt.axhline(avg_duration, color='blue', linewidth=2)
    plt.xlabel('Episode Index')
    plt.ylabel('Navigation Duration [s]')
    plt.title("Navigation Duration")
    plt.savefig(log_path + '/nav_duration.png')

    avg_path_length = get_successful_avg(path_length_data, success_data)
    plt.figure(3)
    plt.bar(label_x, path_length_data, color=color_array)
    plt.axhline(avg_path_length, color='blue', linewidth=2)
    plt.xlabel('Episode Index')
    plt.ylabel('Navigation Path Length [m]')
    plt.title("Navigation Path Length")
    plt.savefig(log_path + '/nav_path_length.png')

'''
DESCRIPTION: TODO...
'''
def euclidean_distance(pos1, pos2):
    return np.sqrt( pow((pos1.x - pos2.x), 2) + pow((pos1.y - pos2.y), 2) + pow((pos1.z - pos2.z), 2))

'''
DESCRIPTION: TODO...
'''
prev_pos = Point()
total_distance_episode = 0.0
def getDistance(msg):
    global prev_pos, total_distance_episode
    dist = euclidean_distance(msg.pose.pose.position, prev_pos)
    total_distance_episode += dist
    prev_pos = msg.pose.pose.position

'''
DESCRIPTION: TODO...
'''
gr = 0.0
def getGoalReachingStatus(msg):
    global gr
    if msg.data:
        gr = 1.0
    else:
        gr = 0.0

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
    NUA TODO: Update Description!
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80) # type: ignore

    rospack = rospkg.RosPack()
    mobiman_path = rospack.get_path('mobiman_simulation') + "/"

    ## Initialize the parameters
    igibson_config_file = rospy.get_param('igibson_config_file', "")
    rl_algorithm = rospy.get_param('rl_algorithm', "")
    data_path = rospy.get_param('drl_data_path', "")
    initial_training_path = rospy.get_param('initial_training_path', "")
    initial_training_model_name = rospy.get_param('initial_training_model_name', "")
    testing_benchmark_name = rospy.get_param('testing_benchmark_name', "")
    n_testing_eval_episodes = rospy.get_param('n_testing_eval_episodes', "")
    #plot_title = rospy.get_param('plot_title', "")
    #plot_moving_average_window_size_timesteps = rospy.get_param('plot_moving_average_window_size_timesteps', 0)
    #plot_moving_average_window_size_episodes = rospy.get_param('plot_moving_average_window_size_episodes', 0)

    print("[drl_testing_sb3_mobiman_jackalJaco::main] igibson_config_file: " + str(igibson_config_file))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] rl_algorithm: " + str(rl_algorithm))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] data_path: " + str(data_path))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] initial_training_path: " + str(initial_training_path))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] initial_training_model_name: " + str(initial_training_model_name))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] testing_benchmark_name: " + str(testing_benchmark_name))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] n_testing_eval_episodes: " + str(n_testing_eval_episodes))
    #print("[drl_testing_sb3_mobiman_jackalJaco::main] plot_title: " + str(plot_title))
    #print("[drl_testing_sb3_mobiman_jackalJaco::main] plot_moving_average_window_size_timesteps: " + str(plot_moving_average_window_size_timesteps))
    #print("[drl_testing_sb3_mobiman_jackalJaco::main] plot_moving_average_window_size_episodes: " + str(plot_moving_average_window_size_episodes))

    ## Create the folder name that the data is kept
    #data_file_tag = createFileName()
    #data_folder_tag = data_file_tag + "_" + rl_algorithm + "_mobiman" # type: ignore
    #data_name = data_folder_tag + "/" # type: ignore
    data_folder_path = mobiman_path + data_path
    if initial_training_path:
        data_folder_path += initial_training_path
    #else:
    #    data_folder_path += "benchmark/"

    print("[drl_testing_sb3_mobiman_jackalJaco::main] data_folder_path: " + str(data_folder_path))

    #os.makedirs(data_folder_path, exist_ok=True)

    #new_trained_model_file = data_folder_path + "trained_model"
    testing_log_file_name = "testing_log_" + testing_benchmark_name + "_" + initial_training_model_name + ".csv"
    testing_log_file = data_folder_path + testing_log_file_name
    tensorboard_log_path = data_folder_path + rl_algorithm + "_testing_tensorboard/"

    ## Keep all parameters in an array to save
    testing_log_data = []
    testing_log_data.append(["igibson_config_file", igibson_config_file])
    testing_log_data.append(["rl_algorithm", rl_algorithm])
    testing_log_data.append(["data_path", data_path])
    testing_log_data.append(["initial_training_path", initial_training_path])
    testing_log_data.append(["initial_training_model_name", initial_training_model_name])
    testing_log_data.append(["testing_benchmark_name", testing_benchmark_name])
    testing_log_data.append(["n_testing_eval_episodes", n_testing_eval_episodes])
    #testing_log_data.append(["plot_title", plot_title])
    #testing_log_data.append(["plot_moving_average_window_size_timesteps", plot_moving_average_window_size_timesteps])
    #testing_log_data.append(["plot_moving_average_window_size_episodes", plot_moving_average_window_size_episodes])

    ## Write all parameters into the log file of the training
    write_data(data_folder_path, testing_log_file_name, testing_log_data)

    #print("[drl_testing_sb3_mobiman_jackalJaco::main] DEBUG_INF")
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

    print("[drl_testing_sb3_mobiman_jackalJaco::main] ros_path: " + str(igibson.ros_path))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] config_file: " + str(igibson_config_file))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] config_path: " + str(igibson_config_path))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] config_data: " + str(igibson_config_data))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] pybullet_mode: " + str(pybullet_mode))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] action_timestep: " + str(action_timestep))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] physics_timestep: " + str(physics_timestep))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] use_pb_gui: " + str(use_pb_gui))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] objects: " + str(objects))
    print("[drl_testing_sb3_mobiman_jackalJaco::main] tensorboard_log_dir: " + str(tensorboard_log_dir))

    #print("[drl_testing_sb3_mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    # Create environments
    print("[drl_testing_sb3_mobiman_jackalJaco::main] BEFORE iGibsonEnv")
    env = iGibsonEnv(
        config_file=igibson_config_path,
        mode=pybullet_mode,
        drl_mode="testing",
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        init_ros_node=True,
        ros_node_id=0,
        use_pb_gui=use_pb_gui,
        automatic_reset=True,
        data_folder_path=data_folder_path,
        log_file=testing_log_file,
        objects=objects,
        flag_drl=True)
    print("[drl_testing_sb3_mobiman_jackalJaco::main] AFTER iGibsonEnv")

    #env = Monitor(env)

    #print("[drl_testing_sb3_mobiman_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    if initial_training_model_name == "":
        initial_trained_model_path = mobiman_path + data_path + initial_training_path + "trained_model" # type: ignore
    else:
        initial_trained_model_path = mobiman_path + data_path + initial_training_path + initial_training_model_name # type: ignore
    
    if rl_algorithm == "SAC":
        model = SAC.load(initial_trained_model_path, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
    
    elif rl_algorithm == "DDPG":
        model = DDPG.load(initial_trained_model_path, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
    
    elif rl_algorithm == "A2C":
        model = A2C.load(initial_trained_model_path, env=env, tensorboard_log=tensorboard_log_path) # type: ignore

    elif rl_algorithm == "DQN":
        model = DQN.load(initial_trained_model_path, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
    
    else:
        model = PPO.load(initial_trained_model_path, env=env, tensorboard_log=tensorboard_log_path) # type: ignore
    #model.set_env(env)
    
    print("[drl_testing_sb3_mobiman_jackalJaco::main] Loaded initial_trained_model_path: " + initial_trained_model_path)

    # Evaluate the policy after training
    start_testing = time.time()
    while(not env.flag_testing_done):
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print("[drl_testing_sb3_mobiman_jackalJaco::main] Testing eval " + str(env.testing_eval_idx) + " and state " + str(env.testing_idx))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_testing_eval_episodes)
        #print("[drl_testing_sb3_mobiman_jackalJaco::main] Mean reward: " + str(mean_reward))
        #print("[drl_testing_sb3_mobiman_jackalJaco::main] Std reward: +/-" + str(std_reward))
        #print("")
        #print("")
        #print("----------")
    end_testing = time.time()
    testing_time = (end_testing - start_testing) / 60

    ## Write all results into the log file of the training
    testing_log_data = []
    testing_log_data.append(["testing_time[min]", testing_time])    
    write_data(data_folder_path, testing_log_file_name, testing_log_data)

    print("[drl_testing_sb3_mobiman_jackalJaco::main] End of testing!")
    print("[drl_testing_sb3_mobiman_jackalJaco::main] testing_time[min]: " + str(testing_time))
    print("")

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

    '''
    ### Start Testing
    # Initialize parameters
    counter = 0
    start_time = rospy.get_time()
    episode_result_list = []
    goal_reached = 0.0
    n_testing_ = 5

    initial_trained_model_path = mobiman_path + data_path + initial_training_path + "trained_model" # type: ignore
    initial_trained_model = initial_trained_model_path
    print("[drl_testing_sb3_mobiman_jackalJaco::main] Loaded initial_trained_model: " + initial_trained_model)

    start_testing = time.time()
    while(not env.flag_testing_done):

        model = PPO.load(initial_trained_model, env=env, tensorboard_log=tensorboard_log_path)
        print("--------------")
        print("[drl_testing_sb3_mobiman_jackalJaco::main] Testing episode " + str(env.testing_idx))
        print("--------------")

        #model.set_env(env)
        obs = env.reset()

        # Evaluate the agent
        episode_reward = 0
        total_distance_episode = 0.0
        start_time = rospy.get_time()

        # if goal_reached == 1.0:
        #     break

        for i in range(max_testing_episode_timesteps):

            #print("--------------")
            #print("[drl_testing_sb3_mobiman_jackalJaco::main] i: {}".format(i))
            #print("--------------")

            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            goal_reached = gr
            
            if (done):
                #print("--------------")
                #print("[drl_testing_sb3_mobiman_jackalJaco::main] Done!")
                #print("--------------")
                counter += 1

                total_time_episode = rospy.get_time() - start_time
                    
                print("--------------")
                print("[drl_testing_sb3_mobiman_jackalJaco::main] goal_reached: {}".format(goal_reached))
                print("[drl_testing_sb3_mobiman_jackalJaco::main] total_time_episode: {}".format(total_time_episode))
                print("[drl_testing_sb3_mobiman_jackalJaco::main] total_distance_episode: {}".format(total_distance_episode))
                print("--------------")

                episode_result_list.append([goal_reached, total_time_episode, total_distance_episode])
                break
            
            else:
                if i == max_testing_episode_timesteps-1:
                    print("--------------")
                    print("[drl_testing_sb3_mobiman_jackalJaco::main] Max number of timesteps has been reached!")
                    print("--------------")
                    counter += 1

                    total_time_episode = rospy.get_time() - start_time
                    
                    print("--------------")
                    print("[drl_testing_sb3_mobiman_jackalJaco::main] goal_reached: {}".format(goal_reached))
                    print("[drl_testing_sb3_mobiman_jackalJaco::main] total_time_episode: {}".format(total_time_episode))
                    print("[drl_testing_sb3_mobiman_jackalJaco::main] total_distance_episode: {}".format(total_distance_episode))
                    print("--------------")

                    episode_result_list.append([goal_reached, total_time_episode, total_distance_episode])
                    
                    obs = env.reset()
                    break
    end_testing = time.time()

    testing_time = (end_testing - start_testing) / 60

    print("[drl_testing_sb3_mobiman_jackalJaco::main] End of testing!")
    print("[drl_testing_sb3_mobiman_jackalJaco::main] testing_time[min]: " + str(testing_time))
    '''

    '''
    testing_log_data = []
    #testing_log_data.append(["total_training_episodes", total_training_episodes])
    #testing_log_data.append(["total_training_timesteps", total_training_timesteps])
    testing_log_data.append(["learning_time[min]", learning_time])

    print("--------------")
    print("[mobiman_drl_training::__main__] End of training!")
    print("[mobiman_drl_training::__main__] learning_time[min]: " + str(learning_time))
    print("--------------")
    #rospy.logdebug("[mobiman_drl_training::__main__] End of training!")

    ## Write all results into the log file of the training
    write_data(testing_log_file, testing_log_data)
    '''

    '''
    ## Write testing results
    testing_result_log_file = data_folder_path + "testing_result_log.csv"
    result_file = open(testing_result_log_file, 'w')
    with result_file:     
        write = csv.writer(result_file)
        episode_result_list.insert(0, ["success", "duration", "path_length"])
        write.writerows(episode_result_list)
    '''

    ## Save the result plots of the testing
    #plot_testing_result(data_folder_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
