#!/usr/bin/env python3

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.rsg_anymal import NormalSampler
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
import os
import time
import numpy as np
import torch
import argparse
from collections import deque
import math
from datetime import datetime
from raisimGymTorch.algo.sac import SAC
from raisimGymTorch.algo.sac.replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--gamma', type=float, default=0.996, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.02, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
seed = cfg['seed']
# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])

# shortcuts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

saver = ConfigurationSaver(log_dir=home_path + '/raisimGymTorch/data/anymal_locomotion', save_items=[task_path + '/cfg.yaml', task_path + '/Environment.hpp'])
log_dir = os.path.join(saver.data_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update


torch.manual_seed(seed)
np.random.seed(seed)

# Agent
agent = SAC(env.num_obs, env.num_acts, args)
print(agent.policy)

# Memory
memory = ReplayMemory(args.replay_size, seed)
# Training Loop
updates = 0

for update in range(1000000):
    if update % cfg['environment']['eval_every_n'] == 0:
        env.turn_on_visualization()

    if update % cfg['environment']['eval_every_n'] == 1:
        env.turn_off_visualization()

    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    observation = env.observe()

    for step in range(n_steps):
        action = agent.select_action(observation)  # Sample action from policy
        observation_next = env.observe()

        if len(memory) > args.batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

        reward, dones = env.step(action)
        mask = 1.0 - dones

        for st, act, rew, next_st, msk in zip(observation, action, reward, observation_next, mask):
            memory.push(st, act, rew, next_st, msk) # Append transition to memory

        observation = observation_next

        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)

    # take st step to get value obs
    x, y, z = env.getCoords()
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    env.curriculum_callback()
    end = time.time()

    writer.add_scalar('General/dones', average_dones, update)
    writer.add_scalar('General/reward', average_ll_performance, update)
    writer.add_scalar('General/x', x, update)
    writer.add_scalar('General/y', y, update)
    writer.add_scalar('General/z', z, update)
    print("Iteration: ", update, "; Real time factor: ", total_steps/(end-start)*cfg['environment']['control_dt'])


    if update % cfg['environment']['eval_every_n'] == 0:
        agent.save_checkpoint(saver.data_dir + '/full_' + str(update) + '.pt')
        env.save_scaling(saver.data_dir, str(update))

        # print("Visualizing and evaluating the current policy")
        #
        # env.turn_on_visualization()
        # env.start_video_recording(datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        #
        # for step in range(n_steps*2):
        #     with torch.no_grad():
        #         frame_start = time.time()
        #         obs = env.observe(False)
        #         action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        #         reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        #         frame_end = time.time()
        #         wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        #         if wait_time > 0.:
        #             time.sleep(wait_time)
        #
        # env.stop_video_recording()
        # env.turn_off_visualization()
        #
        # env.reset()
        # env.save_scaling(saver.data_dir, str(update))
