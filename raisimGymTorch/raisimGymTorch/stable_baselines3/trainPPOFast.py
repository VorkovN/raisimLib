import gym
import os
import datetime
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper

#ЕСЛИ ИМПОРТ ПЕРЕНЕСТИ НА СТРОЧКУ НИЖЕ ПРОИЗВОДИТЕЛЬНОСТЬ УПАДЕТ В 10 РАЗ!!!
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import HER
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from raisimGymTorch.env.bin import anymal_climbing as rsg_anymal
# from stable_baselines3.common.policies import ActorCriticPolicy
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
batch_size = int(n_steps*cfg['environment']['num_envs']/4)

# if args.environment_type == 'climbing':
#     from raisimGymTorch.env.bin import anymal_climbing as rsg_anymal
# elif args.environment_type == 'obstacle':
#     from raisimGymTorch.env.bin import anymal_obstacle as rsg_anymal
# elif args.environment_type == 'run':
#     from raisimGymTorch.env.bin import anymal_run as rsg_anymal
# elif args.environment_type == 'turn':
#     from raisimGymTorch.env.bin import anymal_turn as rsg_anymal

env = VecEnv(rsg_anymal.RaisimGymEnv(rscPath, dump(cfg['environment'], Dumper=RoundTripDumper)))
env.reset()
# class CustomPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=dict(pi=[128, 128], vf=[128, 128]))

if args.algorithm == 'A2C':
    if not args.model_path:
        print("A2C new algorithm:")
        model = A2C("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("A2C old algorithm:")
        model = A2C.load(args.model_path, env)
elif args.algorithm == 'DDPG':
    if not args.model_path:
        print("DDPG new algorithm:")
        model = DDPG("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("DDPG old algorithm:")
        model = DDPG.load(args.model_path, env)
elif args.algorithm == 'DQN':
    if not args.model_path:
        print("DQN new algorithm:")
        model = DQN("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("DQN old algorithm:")
        model = DQN.load(args.model_path, env)
elif args.algorithm == 'HER':
    if not args.model_path:
        print("HER new algorithm:")
        model = HER("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("HER old algorithm:")
        model = HER.load(args.model_path, env)
elif args.algorithm == 'PPO':
    if not args.model_path:
        print("PPO new algorithm:")
        model = PPO("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.98, clip_range=0.5, clip_range_vf=0.5)
    else:
        print("PPO old algorithm:")
        model = PPO.load(args.model_path, env)
elif args.algorithm == 'SAC':
    if not args.model_path:
        print("SAC new algorithm:")
        model = SAC("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("SAC old algorithm:")
        model = SAC.load(args.model_path, env)
elif args.algorithm == 'TD3':
    if not args.model_path:
        print("TD3 new algorithm:")
        model = TD3("MlpPolicy", env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("TD3 old algorithm:")
        model = TD3.load(args.model_path, env)
else:
    print("Wrong algorithm:")
    exit(0)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        x, y, z = env.getCoords()
        self.logger.record('Coords/x', x)
        self.logger.record('Coords/y', y)
        self.logger.record('Coords/z', z)
        return True


for iteration in range(1000):
    print("iteration: ", iteration, flush=True)
    model.learn(total_timesteps=200000000, progress_bar=True, callback=TensorboardCallback())
    model.save(modelsPath + modelName + str(iteration))


