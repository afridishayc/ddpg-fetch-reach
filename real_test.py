import copy
import math
import os
from collections import namedtuple

import gym
import ipywidgets as widgets
import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

env = gym.make('FetchReach-v1')
torch.cuda.current_device()

if torch.cuda.is_available():
	device = "cuda:0" 
else:
	device = "cpu"


class Actor(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, units=256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.units = units
        
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.units)])
        self.layers.extend([ nn.Linear(self.units, self.units) for i in range(1, self.hidden_layers) ])
        self.layers.append(nn.Linear(self.units, self.output_size))
    
    def forward(self, states):
        vals = states
        for layer_index in range(len(self.layers) - 1):
            vals = F.relu(self.layers[layer_index](vals))
        vals = torch.tanh(self.layers[layer_index + 1](vals))
        return vals



target_actor = Actor(16, 4, 5)
target_actor.to(device)
target_actor.load_state_dict(torch.load("her_ta_6"))
target_actor.eval()

print("Model loaded")


state = env.reset()
for _ in range(1000):
    env.render()
    state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]), axis=0)
    cuda_state = torch.tensor([state], device=device, dtype=torch.float)
    action = target_actor(cuda_state)
    action = action.cpu()
    #print(action.detach().numpy()[0])
    next_state, reward, done, info = env.step(action.detach().numpy()[0])
    state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]), axis=0)
    state = next_state
    if done:
        print(next_state, reward, done, info)
        break
        
env.close()



# state = env.reset()
# for _ in range(1000):
#     env.render()
#     # state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]), axis=0)
#     cuda_state = torch.tensor([state], device=device, dtype=torch.float)
#     action = target_actor(cuda_state)
#     action = action.cpu()
#     #print(action.detach().numpy()[0])
#     next_state, reward, done, info = env.step(action.detach().numpy()[0])
#     # state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]), axis=0)
#     state = next_state
#     if done:
#         print(next_state, reward, done, info)
#         break
        
# env.close()