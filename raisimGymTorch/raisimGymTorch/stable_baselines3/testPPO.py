import gym
import os
import datetime
import time

from ruamel.yaml import YAML, dump, RoundTripDumper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from raisimGymTorch.env.bin import rsg_anymal
from raisimGymTorch.stable_baselines3.RaisimSbGymVecEnv import RaisimSbGymVecEnv as VecEnv


# Parallel environments
# directories
stb_path = os.path.dirname(os.path.realpath(__file__))
rsc_path = stb_path + "/../../../rsc"
task_path = stb_path + "/../env/envs/rsg_anymal"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1
env = VecEnv(rsg_anymal.RaisimGymEnv(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper)), normalize_ob=True)
obs = env.reset()

n_steps = int(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
model = PPO.load("baselineModels/ppo_anymal0", env)


for iteration in range(1000):
    print("iteration: ", iteration, flush=True)
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(iteration)+'.mp4')

    for step in range(n_steps*2):
            print("step: ", step)
            frame_start = time.time()
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(obs)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()