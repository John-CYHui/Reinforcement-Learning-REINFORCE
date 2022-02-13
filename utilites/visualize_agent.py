import numpy as np
import pickle as pk
import os, sys, inspect
import gym

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from save_frame import *
from agent import *

# Load agent
episode = 10000
with open (f'../results/agent_episode_{episode}.pk', 'rb') as f:
    agent = pk.load(f)

# Init environment
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

frames = []

state = env.reset()
terminal = False
max_t = 1000
t = 0
while not terminal and t <= max_t:
    frames.append(env.render(mode="rgb_array"))
    action = agent.agent_step(state)
    state, reward, terminal, info = env.step(action)
    t += 1
    if terminal:
        break
    
save_frames_as_gif(frames, path='../readme_gif/', filename= f'mountain_car_agent_eps{episode}.gif')