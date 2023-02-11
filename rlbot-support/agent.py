
import numpy as np
from stable_baselines3 import PPO
import pathlib
from action_parser import LookupAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.0001,
            "clip_range": .02,
            "n_envs": 1,
            "net_arch":[0, dict(pi=[512, 512, 256], vf=[512, 512, 400, 256])],
        }
        
        self.actor = PPO.load(str(_path) + '/Europa1', device='cpu', custom_objects=custom_objects)
        self.parser = LookupAction()


    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")