import numpy as np
from actions import *
import time
import os
import math


class Agent:
    def __init__(self, rob):
        """
        Each state has multiple Q-values, therefore q_values is dict() object with keys=state and values=list of q-vals
        num_states likely equals 6: 5 observing states and one state when nothing is observed
        """
        self.gamma = 0.9
        self.eps = 0.5
        self.epsmin = 0.1
        self.decay = 0.05
        self.alpha = 0.7
        self.rob = rob
        self.q_values = {x: [0, 0, 0] for x in range(4)}    # for now assuming five actions
        self.current_state = None   # state agent is in
        self.observed_state = None  # state agent is in after taking action a
        self.last_action = None
        self.threshold = 0.15
        self.total_distance = 0
        self.terminal_state = False
        self.num_moves = 0
        self.pos_before = [None, None]
        self.pos_after = [None, None]


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: observed front, right, left, front&right, front&left, nothing
        returns int
        """
        read = self.rob.read_irs()
        sensors = [np.inf if x == False else x for x in read]   #only take front sensors
        self.sensors_triggered = sum([x < self.threshold for x in read])
        if all([x == np.inf for x in sensors]):
            return 0    # state 0 is no observations

        elif np.argmin(sensors) == 3 or np.argmin(sensors) == 4:
            if sensors[3] < self.threshold or sensors[4] < self.threshold:
                return 1       # object to the right

        elif np.argmin(sensors) == 5:
            if sensors[5] < self.threshold:
                return 2      # object directly in front

        elif np.argmin(sensors) == 6 or np.argmin(sensors) == 7:
            if sensors[6] < self.threshold or sensors[7] < self.threshold:
                return 3      # object to the left

        # elif np.argmin(sensors) in [0, 1, 2]:
        #     if any([x < self.threshold for x in sensors[0:2]]) and any([x < self.threshold for x in sensors[3:]]):
        #         return 4    # object in back and in front

        return 0


    def get_position(self):
        position = self.rob.getPosition()[1]
        return [position[0], position[1]]


    def update_distance(self):
        xdiff = abs(self.pos_before[0] - self.pos_after[0])
        ydiff = abs(self.pos_before[1] - self.pos_after[1])
        distance = math.sqrt(xdiff**2 + ydiff**2)
        self.total_distance += distance


    def action(self, state):
        """
        Use epsilon greedy policy to determine action
        runs certain action based on index, currently 5 actions
        """
        action_index = np.argmax(self.q_values[state])
        if np.random.binomial(1, self.eps) == 1:
            action_index = np.random.choice([x for x in range(3) if x != action_index])  #pick a random action based on the index. see actions.py
        self.last_action = action_index
        self.num_moves += 1
        select_action(self.rob, action_index)


    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        if self.total_distance > 0.6:
            self.terminal_state = True
            return 50
        if self.observed_state == 0:    # no object in proximity
            return -1
        else:
            return -1



    def calc_Q_values(self, action, reward):
        """
        update q_values
        """
        state = self.current_state
        _state = self.observed_state
        current_q = self.q_values[state][action]
        next_q = np.max(self.q_values[_state])
        a = self.alpha
        g = self.gamma
        self.q_values[state][action] = current_q + a*(reward + g*next_q - current_q)



def train_loop(rob, episodes=10, steps=200):
    """
    Combines all of the above to run a training loop and update the Q-values
    Does 15 training epochs with 50 steps per epoch
    returns nothing, should likely store values of self.q_values in file
    """
    agent = Agent(rob)
    for episode in range(episodes):
        agent.rob.play_simulation()
        time.sleep(3)
        agent.current_state = agent.get_state()
        agent.total_distance = 0
        for step in range(steps):
            if agent.terminal_state:
                agent.terminal_state = False
                break
            agent.pos_before = agent.get_position()
            agent.action(agent.current_state)
            time.sleep(1)
            agent.pos_after = agent.get_position()
            agent.update_distance()
            print(agent.current_state)
            print(agent.total_distance)
            agent.observed_state = agent.get_state()
            reward = agent.get_reward()
            agent.calc_Q_values(agent.last_action, reward)
            agent.current_state = agent.observed_state

        if agent.eps > agent.epsmin:
            agent.eps -= agent.decay

        for key, values in agent.q_values.items():
            print(f"State {key} Q-values: {values}")
        agent.rob.stop_world()
        time.sleep(1)


    if os.path.exists("./src/Qvalues.txt"):
        os.remove('./src/Qvalues.txt')
    for key, values in agent.q_values.items():
        f = open('./src/Qvalues.txt', 'a')
        string = ""
        for value in values:
            string = string + "," + str(value)
        f.write(string)
        f.close()

