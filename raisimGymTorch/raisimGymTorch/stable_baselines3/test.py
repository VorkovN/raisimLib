import gym
import os
import datetime
import time
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper

#ЕСЛИ ИМПОРТ ПЕРЕНЕСТИ НА СТРОЧКУ НИЖЕ ПРОИЗВОДИТЕЛЬНОСТЬ УПАДЕТ В 10 РАЗ!!!

from sb3_contrib import RecurrentPPO as RPPO
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from raisimGymTorch.stable_baselines3.RaisimSbGymVecEnv import RaisimSbGymVecEnv as VecEnv
from stable_baselines3.common.callbacks import BaseCallback

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
batch_size = int(n_steps*cfg['environment']['num_envs'])
model_load_path = os.path.join(args.model_path)

if args.algorithm == 'PPO':
    from stable_baselines3.ppo.policies import MlpPolicy
elif args.algorithm == 'RPPO':
    from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy as MlpPolicy
elif args.algorithm == 'TRPO':
    from sb3_contrib.trpo.policies import MlpPolicy
elif args.algorithm == 'SAC':
    from stable_baselines3.sac.policies import MlpPolicy, SACPolicy, MultiInputPolicy
elif args.algorithm == 'TD3':
    from stable_baselines3.td3.policies import MlpPolicy
else:
    print("Wrong algorithm:")
    exit(0)

if args.environment_type == 'climbing':
    from raisimGymTorch.env.bin import anymal_climbing as rsg_anymal
elif args.environment_type == 'obstacle':
    from raisimGymTorch.env.bin import anymal_obstacle as rsg_anymal
elif args.environment_type == 'run':
    from raisimGymTorch.env.bin import anymal_run as rsg_anymal
elif args.environment_type == 'turn':
    from raisimGymTorch.env.bin import anymal_turn as rsg_anymal

env = VecEnv(rsg_anymal.RaisimGymEnv(rscPath, dump(cfg['environment'], Dumper=RoundTripDumper)))
class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=dict(pi=[128, 128], vf=[128, 128]))

class CustomSACPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, net_arch=[400, 400])

if args.algorithm == 'PPO':
    print("PPO old algorithm:")
    model = PPO.load(model_load_path, env)
elif args.algorithm == 'RPPO':
    print("RPPO old algorithm:")
    model = RPPO.load(model_load_path, env)
elif args.algorithm == 'TRPO':
    print("TRPO old algorithm:")
    model = TRPO.load(model_load_path, env=env)
elif args.algorithm == 'SAC':
    print("SAC old algorithm:")
    model = SAC.load(model_load_path, env)
elif args.algorithm == 'TD3':
    print("TD3 old algorithm:")
    model = TD3.load(model_load_path, env)
else:
    print("Wrong algorithm: " + args.algorithm)
    exit(0)


for iteration in range(3):
   obs = env.reset()
   env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(iteration)+'.mp4')

   for step in range(n_steps):
       frame_start = time.time()
       action, _state = model.predict(obs, deterministic=False)
       obs, reward, done, info = env.step(action)
       frame_end = time.time()
       wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
       if wait_time > 0.:
           time.sleep(wait_time)

   env.stop_video_recording()


