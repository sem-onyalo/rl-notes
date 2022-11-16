from typing import List

import numpy as np
import torch
import torch.nn as nn

from .algorithm import Algorithm

class AlgorithmNet(Algorithm):
    device:torch.device

    def parse_layers_str(self, layers_str:str) -> List[int]:
        min_layers = 2
        layers = list(map(int, layers_str.split(",")))
        assert len(layers) >= min_layers, f"Error: number of layers must be a minimum of {min_layers}"
        return layers

    def init_first_layer(self, layers:List[int]) -> List[nn.Module]:
        net_layers = []
        if layers[0] != len(self.mdp.states):
            net_layers.append(nn.Linear(len(self.mdp.states), layers[0]))

        return net_layers

    def init_middle_layers(self, layers:List[int], net_layers:List[nn.Module], activation_layer_type:str) -> None:
        for i in range(1, len(layers) - 1):
            net_layers.append(nn.Linear(layers[i - 1], layers[i]))
            if activation_layer_type == "batch-norm":
                net_layers.append(nn.BatchNorm1d(layers[i]))
            elif activation_layer_type == "relu":
                net_layers.append(nn.ReLU())
            else:
                raise Exception(f"Error creating middle layer, activation layer type {activation_layer_type} is invalid")

        net_layers.append(nn.Linear(layers[-2], layers[-1]))

    def init_final_layer(self, layers:List[int], net_layers:List[nn.Module]) -> None:
        if layers[-1] != len(self.mdp.actions):
            net_layers.append(nn.Linear(layers[-1], len(self.mdp.actions)))

    def build_linear_function(self, layers_str:str):
        layers = self.parse_layers_str(layers_str)

        net_layers = self.init_first_layer(layers)

        self.init_middle_layers(layers, net_layers, "batch-norm")

        self.init_final_layer(layers, net_layers)

        return nn.Sequential(*net_layers)

    def build_linear_softmax_function(self, layers_str:str):
        layers = self.parse_layers_str(layers_str)

        net_layers = self.init_first_layer(layers)

        self.init_middle_layers(layers, net_layers, "relu")

        self.init_final_layer(layers, net_layers)

        net_layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*net_layers)

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
