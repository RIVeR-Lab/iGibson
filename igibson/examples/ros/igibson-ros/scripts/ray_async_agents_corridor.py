"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""

from time import sleep
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random

os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):

        print("[ray_async_agents::SimpleCorridor::__init__] START")

        self.end_pos = config["corridor_length"]
        self.cur_pos = 0

        self.total_step_num = 0
        self.step_num = 0

        self.worker_index = config.worker_index 
        self.num_workers = config.num_workers

        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

        print("[ray_async_agents::SimpleCorridor::__init__] worker_index: " + str(self.worker_index))
        print("[ray_async_agents::SimpleCorridor::__init__] num_workers: " + str(self.num_workers))

        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=config.worker_index * config.num_workers)

        print("[ray_async_agents::SimpleCorridor::__init__] END")

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        self.step_num = 0
        return [self.cur_pos], {}

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
            #print("[ray_async_agents] worker_index: " + str(self.worker_index))
            #print("[ray_async_agents] total_step_num: " + str(self.total_step_num))
            sleep(1)
            #print("[ray_async_agents] WOKE UP!")
            #print()

        if self.worker_index == 1:
            print("[ray_async_agents] worker_index: " + str(self.worker_index))
            print("[ray_async_agents] total_step_num: " + str(self.total_step_num))
            print()

        if self.total_step_num >= 128:
            print("[ray_async_agents] worker_index: " + str(self.worker_index))
            print("[ray_async_agents] num_workers: " + str(self.num_workers))
            print("[ray_async_agents] total_step_num: " + str(self.total_step_num))
            print("[ray_async_agents] step_num: " + str(self.step_num))
            print("[ray_async_agents] action: " + str(action))
            print()
        
        # Produce a random reward when we reach the goal.
        return (
            [self.cur_pos],
            random.random() * 2 if done else -0.1,
            done,
            truncated,
            {},
        )


if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    config = (
        PPOConfig()
        .environment(SimpleCorridor, env_config={"corridor_length": 5})
        .rollouts(num_rollout_workers=2, remote_worker_envs=False, sample_async=False)
        .resources(num_gpus=1)
        .training(train_batch_size=128)
    )

    stop_iters = 1
    stop_timesteps = 100000 
    stop_reward = 0.1

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    no_tune = True

    if no_tune:
        config.lr = 1e-3
        algo = config.build()

        # run manual training loop and print results after each iteration
        for _ in range(stop_iters):
            result = algo.train()
            print("#######################################")
            print("#######################################")
            print(pretty_print(result))
            print("#######################################")
            print("#######################################")
            print()
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= stop_timesteps
                or result["episode_reward_mean"] >= stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        as_test = True
        if as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, stop_reward)

    ray.shutdown()