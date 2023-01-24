import io
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .policy import Policy
from constants import EPSILON_GREEDY_EXPLORE
from mdp import MDP

POLICY_NAME = "policy-approximator"

class PolicyApproximatorArgs:
    explore_type:str
    epsilon:float
    epsilon_floor:float
    decay_type:str
    decay_rate:int
    change_rate:float

class PolicyApproximator(Policy):
    def __init__(self, mdp:MDP, args:PolicyApproximatorArgs) -> None:
        super().__init__(mdp, POLICY_NAME)

        self.explore_type = args.explore_type
        self.epsilon = args.epsilon
        self.epsilon_floor = args.epsilon_floor
        self.decay_type = args.decay_type
        self.decay_rate = args.decay_rate

        self.model_file_ext = "pth"
        self.loss_function = nn.SmoothL1Loss()
        self.target_function = self.build_function()
        self.behaviour_function = self.build_function()
        self.optimizer = optim.RMSprop(self.behaviour_function.parameters(), lr=args.change_rate)
        self.target_function.load_state_dict(self.behaviour_function.state_dict())
        self.target_function.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"function architecture:\n{self.behaviour_function}")
        
    def __call__(self, state:torch.Tensor) -> int:
        """
        Return the optimal (max) action.
        """
        predictions = self.get_predictions(state)
        return predictions.argmax()

    def choose_action(self, state:torch.Tensor) -> int:
        """
        Choose an action stochastically.
        """
        if self.explore_type == EPSILON_GREEDY_EXPLORE:
            return self.get_epsilon_greedy_action(state)
        else:
            raise Exception(f"{self.explore_type} not yet implemented")

    def get_epsilon_greedy_action(self, state:torch.Tensor) -> int:
        """
        Choose an action using the epsilon-greedy algorithm.
        """
        do_explore = random.random() < self.epsilon
        if do_explore:
            return random.randint(0, self.mdp.n_action - 1)
        else:
            return self.__call__(state)

    def get_value(self, state:torch.Tensor, action:int) -> float:
        """
        Returns the value for a state and action.
        """
        predictions = self.get_predictions(state)
        return predictions[action]

    def get_values(self, state:np.ndarray) -> np.ndarray:
        """
        Returns the values for each action.
        """
        transformed_state = self.transform_state(state)
        predictions = self.get_predictions(transformed_state)
        return predictions.numpy()

    def transform_state(self, state:np.ndarray) -> str:
        """
        Transform a state to the format the action-value function uses.
        """
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def update(self, *args) -> None:
        """
        Updates the action-value function.
        """
        transition = args[0]
        discount_rate = args[1]
        update_target_function = args[2]

        state = torch.tensor(np.atleast_2d(transition[0]), dtype=torch.float32).reshape(self.mdp.d_state)
        action = transition[1]
        next_state = torch.tensor(np.atleast_2d(transition[2]), dtype=torch.float32).reshape(self.mdp.d_state)
        reward = transition[3]

        value = self.behaviour_function(state)[action]
        max_value = self.target_function(next_state).max()
        expected_value = reward + (discount_rate * max_value)

        loss = self.loss_function(expected_value, value)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.behaviour_function.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if update_target_function:
            self.target_function.load_state_dict(self.behaviour_function.state_dict())

    def get_model(self) -> io.BytesIO:
        """
        Gets the model parameters as a bytes-like object.
        """
        buffer = io.BytesIO()
        state_dict = self.behaviour_function.state_dict()
        torch.save(state_dict, buffer)
        return buffer

    def load_model(self, buffer:io.BytesIO) -> None:
        """
        Loads the model parameters from a bytes-like object.
        """
        self.behaviour_function.load_state_dict(torch.load(buffer))
        self.behaviour_function.eval()

    def build_function(self):
        layers_list = []
        layers = self.get_layers()
        for i in range(1, len(layers) - 1):
            layers_list.append(nn.Linear(layers[i - 1], layers[i]))
            layers_list.append(nn.BatchNorm1d(layers[i]))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Flatten(0))
        layers_list.append(nn.Linear(layers[-2]**2, layers[-1]))
        return nn.Sequential(*layers_list)

    def get_layers(self) -> list:
        layers = []
        layers.append(self.mdp.d_state[0])
        layers.append(self.mdp.d_state[0]*2)
        layers.append(self.mdp.d_state[0]*4)
        layers.append(self.mdp.d_state[0]*2)
        layers.append(self.mdp.d_state[0])
        layers.append(self.mdp.n_action)
        self.logger.debug(f"layers: {layers}")
        return layers

    def get_predictions(self, state:torch.Tensor):
        with torch.no_grad():
            self.behaviour_function.eval()
            predictions = self.behaviour_function(state)
            self.logger.debug(f"predictions:\n{predictions}")
            self.behaviour_function.train()
            return predictions
