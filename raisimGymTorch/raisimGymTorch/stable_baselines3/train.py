import gym
import os
import datetime
import time
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper

#ЕСЛИ ИМПОРТ ПЕРЕНЕСТИ НА СТРОЧКУ НИЖЕ ПРОИЗВОДИТЕЛЬНОСТЬ УПАДЕТ В 10 РАЗ!!!
from stable_baselines3.common.vec_env import VecNormalize
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
env = VecNormalize(env)
obs = env.reset()
class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=dict(pi=[128, 128], vf=[128, 128]))

class CustomSACPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, net_arch=[400, 400])

if args.algorithm == 'PPO':
    if not args.model_path:
        print("PPO new algorithm:")
        model = PPO(CustomPolicy, env, n_steps=n_steps, gae_lambda=0.95,  learning_rate=0.0003, verbose=0, batch_size=1000, n_epochs=5, tensorboard_log=logsPath, gamma=0.99, clip_range=0.4, device='cuda:1')
    else:
        print("PPO old algorithm:")
        model = PPO.load(model_load_path, env)
        env = env.load(model_load_path+"Env", env)
elif args.algorithm == 'RPPO':
    if not args.model_path:
        print("RPPO new algorithm:")
        model = RPPO(CustomPolicy, env, n_steps=n_steps, gae_lambda=0.95,  learning_rate=0.0003, verbose=0, batch_size=20000, n_epochs=1, tensorboard_log=logsPath, gamma=0.99, clip_range=0.4, device='cuda:1')
    else:
        print("RPPO old algorithm:")
        model = RPPO.load(model_load_path, env)
        env = env.load(model_load_path+"Env", env)
elif args.algorithm == 'TRPO':
    if not args.model_path:
        print("TRPO new algorithm:")
        model = TRPO(CustomPolicy, env, n_steps=n_steps, gae_lambda=0.95, verbose=0, batch_size=1000, tensorboard_log=logsPath, gamma=0.99, target_kl=0.02, device='cuda:0')
    else:
        print("TRPO old algorithm")
        model = TRPO.load(model_load_path, env=env)
        env = env.load(model_load_path+"Env", env)
elif args.algorithm == 'SAC':
    if not args.model_path:
        print("SAC new algorithm:")
        model = SAC(MlpPolicy, env, verbose=0, ent_coef="auto_0.01", tau=0.002, gradient_steps=1, train_freq=1, buffer_size=5000000, batch_size=5000, tensorboard_log=logsPath, gamma=0.99, device='cuda:0')
    else:
        print("SAC old algorithm:")
        model = SAC.load(model_load_path, env)
        env = env.load(model_load_path+"Env", env)
elif args.algorithm == 'TD3':
    if not args.model_path:
        print("TD3 new algorithm:")
        model = TD3(MlpPolicy, env, verbose=0, train_freq=n_steps, batch_size=batch_size, tensorboard_log=logsPath, gamma=0.995)
    else:
        print("TD3 old algorithm:")
        model = TD3.load(model_load_path, env)
        env = env.load(model_load_path+"Env", env)
else:
    print("Wrong algorithm: " + args.algorithm)
    exit(0)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        x, y, z, r, d = env.getCoords()
        self.logger.record('Coords/x', x)
        self.logger.record('Coords/y', y)
        self.logger.record('Coords/z', z)
        self.logger.record('Stat/reward', r)
        self.logger.record('Stat/done', d)
        return True

    def _on_rollout_end(self) -> None:
        env.reset()
        print(self.num_timesteps)
        pass

for iteration in range(10):
    #    print("iteration: ", iteration, flush=True)
    print(model.get_parameters())
    model.learn(total_timesteps=200000000, progress_bar=True, callback=TensorboardCallback())

    archeiveName = str(modelsPath + modelName + str(iteration))
    envName = archeiveName+"Env"
    model.save(archeiveName)
    env.save(envName)
    del model
    del env

    env = VecEnv(rsg_anymal.RaisimGymEnv(rscPath, dump(cfg['environment'], Dumper=RoundTripDumper)))
    env = VecNormalize(env)
    env = env.load(envName, env)
    obs = env.reset()

    if args.algorithm == 'PPO':
        model = PPO.load(archeiveName, env=env)
    elif args.algorithm == 'RPPO':
        model = RPPO.load(archeiveName, env=env)
    elif args.algorithm == 'TRPO':
        model = TRPO.load(archeiveName, env=env)
    elif args.algorithm == 'SAC':
        model = SAC.load(archeiveName, env=env)
    elif args.algorithm == 'TD3':
        model = TD3.load(archeiveName, env=env)



#    model.save(modelsPath + modelName + str(iteration))
#    policy = model.policy
#    policy.save(modelsPath + modelName + str(iteration) + "Policy")

#    obs = env.reset()
#    for iteration in range(3):
#        print("iteration: ", iteration, flush=True)
#        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(iteration)+'.mp4')
#
#        for step in range(n_steps):
#            print("step: ", step)
#            frame_start = time.time()
#            action, _state = model.predict(obs, deterministic=False)
#            obs, reward, done, info = env.step(action)
#            frame_end = time.time()
#            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
#            if wait_time > 0.:
#                time.sleep(wait_time)
#
#        env.stop_video_recording()

