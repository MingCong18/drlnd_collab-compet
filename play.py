from maddpg import MADDPGAgentTrainer
from unityagents import UnityEnvironment
import numpy as np
import torch
import os
from datetime import datetime
from collections import deque
import argparse

def play(env, brain_name, num_agents, agent, num_episodes=10):
    """Execute policy in specified environment
    Args:
        env: Unity environment object
        brain_name: A string parameter indicating name of brain
        num_agents: An integer representing number of agents
        agent: An instance of agent (DDPGMultiAgent or MADDPGAgentTrainer)
    """
    best_score = -np.inf
    scores = []
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    for episode in range(num_episodes):
        episode_scores = []
        score = np.zeros(num_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += env_info.rewards
            states = next_states
            if np.any(dones):
                break
#
        episode_scores.append(score)
        print("Episode Score: {:.2f}".format(np.mean(score)))
        scores.append(score)
    print('Final Score: {:.2f}'.format(np.mean(scores)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', default="Tennis.app", 
        help='environment name')
    parser.add_argument('--checkpoint_path', type=str, default='/Users/mcong/Desktop/github/drlnd_collab-compet/', 
        help='checkpoint path to load')
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")

    args = parser.parse_args()

    max_t = 1000

    # Unity Env
    env = UnityEnvironment(file_name='Tennis.app')
    # brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    state = env_info.vector_observations
    state_shape = state.shape[1]
    action_size = brain.vector_action_space_size

    agent = MADDPGAgentTrainer(state_shape, action_size, num_agents, random_seed=2, dirname=None, 
        model_path=args.checkpoint_path, eval_mode=True)
    play(env, brain_name, num_agents, agent, num_episodes=args.num_episodes)


if __name__ == "__main__":    
    main()