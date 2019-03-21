from maddpg import MADDPGAgentTrainer
from unityagents import UnityEnvironment
import numpy as np
import torch
import os
from datetime import datetime
from collections import deque


def maddpg(env, brain_name, num_agents, agent, max_t=1000):
    """Train DDPG Agent
    Params    
    ======
        env (object): Unity environment instance
        brain_name (string): name of brain
        num_agents (int): number of agents
        agent (DDPGMultiAgent): agent instance
        max_t (int): number of timesteps in each episode
    """
    episode_rewards = []
    average_rewards = []
    i_episode = 0
    best_score = -np.inf
#
    while True:
        i_episode += 1
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += env_info.rewards
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break
#
        episode_rewards.append(np.max(score))
        current_score = np.mean(episode_rewards[-100:])
        average_rewards.append(current_score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_score), end="")
#
        if current_score > best_score:
            # print('Best score found, old: {}, new: {}'.format(best_score, current_score))
            best_score = current_score
            agent.checkpoint()
#
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_score))
#
        if current_score>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, current_score))
            break


def main():
    max_t = 1000
    model = None

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

    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    dirname = 'runs/{}'.format(now)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    agent = MADDPGAgentTrainer(state_shape, action_size, num_agents, random_seed=48, dirname=dirname, model_path=model)
    maddpg(env, brain_name, num_agents, agent, max_t=max_t)


if __name__ == "__main__":    
    main()