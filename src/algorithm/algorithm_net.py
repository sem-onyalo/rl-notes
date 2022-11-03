import numpy as np
import torch
import torch.nn as nn

from .algorithm import Algorithm

class AlgorithmNet(Algorithm):
    device:torch.device

    def build_linear_function(self, layers_str:str):
        # TODO: if the last layer value <> len(self.mdp.actions)
        # then manually add a final layer with len(self.mdp.actions)
        # number of outputs
        layers = list(map(int, layers_str.split(",")))

        net_layers = []
        for i in range(1, len(layers) - 1):
            net_layers.append(nn.Linear(layers[i - 1], layers[i]))
            net_layers.append(nn.BatchNorm1d(layers[i]))
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*net_layers)

    def build_linear_softmax_function(self, layers_str:str):
        layers = list(map(int, layers_str.split(",")))
        assert len(layers) >= 2, "Error: number of layers must be a minimum of 3"

        net_layers = []
        for i in range(1, len(layers) - 1):
            net_layers.append(nn.Linear(layers[i - 1], layers[i]))
            net_layers.append(nn.ReLU())
        net_layers.append(nn.Linear(layers[-2], layers[-1]))

        # final layer needs to be same size as action space length
        if layers[-1] != len(self.mdp.actions):
            net_layers.append(nn.Linear(layers[-1], len(self.mdp.actions)))

        net_layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*net_layers)

    def state_to_tensor(self, state:object):
        if isinstance(state, int) or isinstance(state, float):
            state = np.expand_dims([state], axis=0)

        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def transitions_to_batches(self, transitions):
        batch = list(zip(*transitions))
        states = batch[0]
        actions = batch[1]
        next_states = batch[2]
        rewards = batch[3]

        return states, actions, next_states, rewards

    def get_total_discounted_rewards(self, rewards, discount_rate):
        total_rewards = [0.] * len(rewards)
        for step in range(0, len(rewards)):
            for future_step in range(step, len(rewards)):
                total_rewards[step] = total_rewards[step] + discount_rate**future_step * rewards[future_step]

        return total_rewards
