
import gym
from gym import wrappers
import numpy as np
import torch
import tqdm

import sys
sys.path.append('/Users/zhangxian/Workspace/berkely-rl/homework_fall2023/hw3')
from cs285.infrastructure import pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent

from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_config

CONFIG_FILE = "experiments/dqn/lunarlander.yaml"
GPU_ID = None
LOG_DIR = ""

def main():
    config = make_config(CONFIG_FILE)
    # logger = make_logger(LOG_DIR, config)

    ptu.init_gpu(use_gpu=(GPU_ID is not None), gpu_id=GPU_ID)
    env = config["make_env"]()
    
    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    replay_buffer = ReplayBuffer()
    observation, info = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = 0.1
        
        action = agent.get_action(observation, epsilon)

        next_observation, reward, _, done, info = env.step(action)

        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)
        replay_buffer.insert(observation=observation, 
                             action=action, 
                             reward=reward, 
                             next_observation=next_observation, 
                             done=done)

        # Handle episode termination
        if done:
            observation, info = env.reset()
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)

            update_info = agent.update(batch["observations"], 
                                       batch["actions"], 
                                       batch["rewards"], 
                                       batch["next_observations"], 
                                       batch["dones"], 
                                       step)


if __name__ == "__main__":
    main()
