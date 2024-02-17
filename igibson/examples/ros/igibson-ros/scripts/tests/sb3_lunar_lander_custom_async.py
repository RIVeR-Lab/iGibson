#!/usr/bin/python3

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from lunar_lander_custom import LunarLander

from time import sleep
import numpy as np
import random

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

print("[sb3_async_example::SimpleCorridor::__init__] LIBRARIES LOADED")

class SimpleCorridor(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        #self.observation_space = spaces.Box(low=0, high=255,
        #                                    shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        
        print("[sb3_async_example::SimpleCorridor::__init__] START")

        self.end_pos = config["corridor_length"]
        self.cur_pos = 0

        self.total_step_num = 0
        self.step_num = 0

        self.worker_index = config.worker_index 
        self.num_workers = config.num_workers

        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

        print("[sb3_async_example::SimpleCorridor::__init__] worker_index: " + str(self.worker_index))
        print("[sb3_async_example::SimpleCorridor::__init__] num_workers: " + str(self.num_workers))

    def step(self, action):
        self.total_step_num += 1
        self.step_num += 1

        if self.worker_index == 1:
            sleep(1)

        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = truncated = self.cur_pos >= self.end_pos
        
        if self.worker_index == 2:
            #print("[sb3_async_example] worker_index: " + str(self.worker_index))
            #print("[sb3_async_example] total_step_num: " + str(self.total_step_num))
            sleep(1)
            #print("[sb3_async_example] WOKE UP!")
            #print()

        if self.worker_index == 1:
            print("[sb3_async_example] worker_index: " + str(self.worker_index))
            print("[sb3_async_example] total_step_num: " + str(self.total_step_num))
            print()

        if self.total_step_num >= 128:
            print("[sb3_async_example] worker_index: " + str(self.worker_index))
            print("[sb3_async_example] num_workers: " + str(self.num_workers))
            print("[sb3_async_example] total_step_num: " + str(self.total_step_num))
            print("[sb3_async_example] step_num: " + str(self.step_num))
            print("[sb3_async_example] action: " + str(action))
            print()
        
        # Produce a random reward when we reach the goal.
        return (
            [self.cur_pos],
            random.random() * 2 if done else -0.1,
            done,
            truncated,
            {},
        )

    def reset(self, seed=None, options=None):
        #random.seed(seed)
        self.cur_pos = 0
        self.step_num = 0
        return [self.cur_pos], {}

    def render(self):
        ...

    def close(self):
        ...

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        print("[sb3_async_example] BEFORE register")
        gym.envs.register(
            id='LunarLanderCustom-v0',
            entry_point='lunar_lander_custom:LunarLander',
            max_episode_steps=1000,
            kwargs={'eid' : rank},
        )
        print("[sb3_async_example] AFTER register")

        print("[sb3_async_example] env_id: " + str(env_id))

        env = gym.make(env_id, render_mode="human")
        #env = LunarLander(eid=rank)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def make_env2(rank: int, seed: int = 0):
        def _init() -> SimpleCorridor:
            env = SimpleCorridor()
            env.reset(seed + rank) # type: ignore
            return env

        set_random_seed(seed)
        return _init

# Create environment
# env = gym.make("LunarLander-v2", render_mode="human")

'''
train_env = SubprocVecEnv(
            [make_env("LunarLander-v2", i) for i in range(2)],
            start_method="fork",
        )
'''

train_env = SubprocVecEnv(
            [make_env("LunarLanderCustom-v0", i) for i in range(2)],
            start_method="fork",
        )

'''
# Set multi-process
print("[sb3_async_example::SimpleCorridor::__init__] BEFORE SubprocVecEnv")
num_environments = 2
train_env = SubprocVecEnv([make_env2(i) for i in range(num_environments)])
'''

# Instantiate the agent
print("[sb3_async_example::SimpleCorridor::__init__] BEFORE PPO")
model = PPO("MlpPolicy", train_env, verbose=1)

#print("[sb3_async_example::SimpleCorridor::__init__] DEBUG_INF")
#while(1):
#    continue

# Train the agent and display a progress bar
print("[sb3_async_example::SimpleCorridor::__init__] BEFORE learn")
model.learn(total_timesteps=int(2e5), progress_bar=True)

'''
# Save the agent
model.save("ppo_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
model = PPO.load("ppo_lunar", env=train_env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
'''