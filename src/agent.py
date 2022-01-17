import numpy as np
from actions import *


class Agent:
    def __init__(self, rob):
        """
        Each state has multiple Q-values, therefore q_values is dict() object with keys=state and values=list of q-vals
        num_states likely equals 6: 5 observing states and one state when nothing is observed
        """
        self.gamma = None
        self.eps = 0.5  # high epsilon for testing purposes. subject to change
        self.alpha = None
        self.rob = rob
        self.q_values = {x: [0, 0, 0] for x in range(6)}    # for now assuming three actions


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: observed front, right, left, front&right, front&left, nothing
        returns int
        """
        pass


    def action(self, state):
        """
        Use epsilon greedy policy to determine action
        runs certain action based on index, currently only three actions
        """
        if np.random.binomial(1, self.eps) == 1:
            action_index = np.random.choice([0, 1, 2])  #pick a random action based on the index. see actions.py
        else:
            action_index = np.argmax(self.q_values[state])  # pick the best action based on q-values
        select_action(self.rob, action_index)


    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        pass


    def calc_Q_values(self):
        """
        Use formula to update self.q_values
        returns nothing
        """
        pass


    def train_loop(self, episodes=15, steps=100):
        """
        Combines all of the above to run a training loop and update the Q-values
        Does 15 training epochs with 100 steps per epoch
        returns nothing, should likely store values of self.q_values in file
        """
        pass


