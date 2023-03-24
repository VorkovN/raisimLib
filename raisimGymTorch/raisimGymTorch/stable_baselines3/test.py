import sys
import os
import datetime
import time
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper

from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DQN

from raisimGymTorch.stable_baselines3.RaisimSbGymVecEnv import RaisimSbGymVecEnv as VecEnv

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, default='PPO')
parser.add_argument('-e', '--environment_type', type=str, default='climbing')
parser.add_argument('-p', '--model_path', type=str, default='')
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, default='PPO')
parser.add_argument('-p', '--model_path', type=str, default='')
parser.add_argument('-e', '--environment_type', type=str, default='climbing')
args = parser.parse_args()

stbPath = os.path.dirname(os.path.realpath(__file__))
rscPath = stbPath + "/../../../rsc"
taskPath = stbPath + "/../env/envs/anymal_" + args.environment_type
baselineDataPath = "baselineData/" + str(datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
modelsPath = baselineDataPath + "/baselineModels/"
modelName = "anymal"
logsPath = baselineDataPath

cfg = YAML().load(open(taskPath + "/cfg.yaml", 'r'))
n_steps = int(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
batch_size = int(n_steps*cfg['environment']['num_envs']/4)
cfg['environment']['num_envs'] = 1

if args.environment_type == 'climbing':
    from raisimGymTorch.env.bin import anymal_climbing as rsg_anymal
elif args.environment_type == 'obstacle':
    from raisimGymTorch.env.bin import anymal_obstacle as rsg_anymal
elif args.environment_type == 'run':
    from raisimGymTorch.env.bin import anymal_run as rsg_anymal
elif args.environment_type == 'turn':
    from raisimGymTorch.env.bin import anymal_turn as rsg_anymal

env = VecEnv(rsg_anymal.RaisimGymEnv(rscPath, dump(cfg['environment'], Dumper=RoundTripDumper)))

if args.algorithm == 'PPO':
        print("PPO old algorithm:")
        model = PPO.load(args.model_path, env)
elif args.algorithm == 'TRPO':
        print("TRPO old algorithm:")
        model = TRPO.load(args.model_path, env)
elif args.algorithm == 'SAC':
        print("SAC old algorithm:")
        model = SAC.load(args.model_path, env)
elif args.algorithm == 'TD3':
        print("TD3 old algorithm:")
        model = TD3.load(args.model_path, env)
elif args.algorithm == 'DQN':
        print("DQN old algorithm:")
        model = DQN.load(args.model_path, env)
else:
    print("Wrong algorithm:")
    exit(0)

for iteration in range(500):
    print("iteration: ", iteration, flush=True)
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(iteration)+'.mp4')

    obs = env.reset()
    for step in range(n_steps):
            print("step: ", step)
            frame_start = time.time()
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()