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


if args.algorithm == 'A2C':
    from stable_baselines3.a2c.policies import MlpPolicy
elif args.algorithm == 'DDPG':
    from stable_baselines3.ddpg.policies import MlpPolicy
elif args.algorithm == 'DQN':
    from stable_baselines3.dqn.policies import MlpPolicy
# elif args.algorithm == 'HER':
#     from stable_baselines3.her.policies import MlpPolicy
elif args.algorithm == 'PPO':
    from stable_baselines3.ppo.policies import MlpPolicy
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
env.reset()
class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=dict(pi=[128, 128], vf=[128, 128]))

class CustomSACPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, net_arch=[128, 128])

if args.algorithm == 'A2C':
    if not args.model_path:
        print("A2C new algorithm:")
        model = A2C(CustomPolicy, env, n_steps=n_steps, verbose=0, tensorboard_log=logsPath, gamma=0.98)
    else:
        print("A2C old algorithm:")
        model = A2C.load(args.model_path, env)
elif args.algorithm == 'DDPG':
    if not args.model_path:
        print("DDPG new algorithm:")
        model = DDPG(CustomPolicy, env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("DDPG old algorithm:")
        model = DDPG.load(args.model_path, env)
elif args.algorithm == 'DQN':
    if not args.model_path:
        print("DQN new algorithm:")
        model = DQN(CustomPolicy, env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("DQN old algorithm:")
        model = DQN.load(args.model_path, env)
elif args.algorithm == 'HER':
    if not args.model_path:
        print("HER new algorithm:")
        model = HER(CustomPolicy, env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.97)
    else:
        print("HER old algorithm:")
        model = HER.load(args.model_path, env)
elif args.algorithm == 'PPO':
    if not args.model_path:
        print("PPO new algorithm:")
        model = PPO(CustomPolicy, env, n_steps=n_steps, verbose=0, batch_size=batch_size, n_epochs=4, tensorboard_log=logsPath, gamma=0.99, clip_range=0.3, clip_range_vf=0.3)
    else:
        print("PPO old algorithm:")
        model = PPO.load(args.model_path, env)
elif args.algorithm == 'SAC':
    if not args.model_path:
        print("SAC new algorithm:")
        model = SAC(CustomSACPolicy, env, verbose=0, train_freq=n_steps, ent_coef=2, batch_size=batch_size, tensorboard_log=logsPath, gamma=0.99)
    else:
        print("SAC old algorithm:")
        model = SAC.load(args.model_path, env)
elif args.algorithm == 'TD3':
    if not args.model_path:
        print("TD3 new algorithm:")
        model = TD3(MlpPolicy, env, verbose=0, train_freq=n_steps, batch_size=batch_size, tensorboard_log=logsPath, gamma=0.98)
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

    def _on_rollout_end(self) -> None:
        env.reset()
        pass


for iteration in range(1000):
    print("iteration: ", iteration, flush=True)
    print(model.get_parameters())
    model.learn(total_timesteps=100000000, progress_bar=True, callback=TensorboardCallback())
    model.save(modelsPath + modelName + str(iteration))


