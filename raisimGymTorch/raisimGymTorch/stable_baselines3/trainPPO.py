import gym
import os
import datetime
import time
from ruamel.yaml import YAML, dump, RoundTripDumper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from raisimGymTorch.env.bin import rsg_anymal
from raisimGymTorch.stable_baselines3.RaisimSbGymVecEnv import RaisimSbGymVecEnv as VecEnv
from stable_baselines3.common.callbacks import BaseCallback


stbPath = os.path.dirname(os.path.realpath(__file__))
rscPath = stbPath + "/../../../rsc"
taskPath = stbPath + "/../env/envs/rsg_anymal"
baselineDataPath = "baselineData/" + str(datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
modelsPath = baselineDataPath + "./baselineModels/"
modelName = "ppo_anymal"
logsPath = baselineDataPath# + "/logs/"

cfg = YAML().load(open(taskPath + "/cfg.yaml", 'r'))
n_steps = int(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
batch_size = int(n_steps*cfg['environment']['num_envs']/4)

env = VecEnv(rsg_anymal.RaisimGymEnv(rscPath, dump(cfg['environment'], Dumper=RoundTripDumper)))
env.reset()

model = PPO(MlpPolicy,
            env,
            n_steps=n_steps,
            verbose=0,
            batch_size=batch_size,
            n_epochs=4,
            tensorboard_log=logsPath,
            gamma=0.97)

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
    model.learn(total_timesteps=50000000, progress_bar=True, callback=TensorboardCallback())
    model.save(modelsPath + modelName + str(iteration))


