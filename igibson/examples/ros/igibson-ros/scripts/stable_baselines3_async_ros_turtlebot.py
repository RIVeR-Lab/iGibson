#!/usr/bin/python3

import logging
from logging import config
#import os
#from tracemalloc import stop
#from typing import Callable
import random
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

#import igibson
from igibson.envs.igibson_env import iGibsonEnv

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0.0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=config.worker_index * config.num_workers)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0.0
        return [self.cur_pos], {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = truncated = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return (
            [self.cur_pos],
            random.random() * 2 if done else -0.1,
            done,
            truncated,
            {},
        )

'''
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
'''

"""
Example training code using stable-baselines3 PPO for PointNav task.
"""

'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
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
'''

def main(selection="user", headless=False, short_exec=False):
    """
    Example to set a training process with RAY-RLLib
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    print("[stable_baselines_async_ros_turtlebot::main] START")

    ray.init()

    '''
    config_file = "turtlebot_nav.yaml"
    config_iGibson={
        "config_file": config_file,
        "mode": "headless",
        "action_timestep": 1 / 10.0,
        "physics_timestep": 1 / 120.0,
        "ros_node_id": 0
    }
    '''

    # Set the maximum number of timesteps
    stop_criteria = {
        "timesteps_total": 100,  # Replace with your desired number of timesteps
    }

    '''
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
        .environment(env=iGibsonEnv, env_config=config_iGibson)
        .build()
    )
    '''

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
        .environment(env=SimpleCorridor, env_config={"corridor_length": 5})
        .build()
    )

    tune_flag = False
    ## Run training
    if not tune_flag:
        print("[stable_baselines_async_ros_turtlebot::main] NO TUNING!")
        algo.lr = 1e-3
        
        # run manual training loop and print results after each iteration
        for _ in range(1000):
            result = algo.train()
            
            print(pretty_print(result))
            
            # stop training of the target train steps or reward are reached
            if (result["timesteps_total"] >= stop_criteria["timesteps_total"]):
                break
        algo.stop()
    else:
        print("[stable_baselines_async_ros_turtlebot::main] WITH TUNING!")

        tuner = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop=stop_criteria),
        )
        results = tuner.fit()

    print("[stable_baselines_async_ros_turtlebot::main] END")

    ray.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()