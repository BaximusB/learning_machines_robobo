import numpy as np
from actions import *
import time
import math
import os
import matplotlib.pyplot as plt

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
        self.alpha = 0.4
        self.rob = rob
        self.q_values = {x: [0, 0, 0, 0, 0] for x in range(7)}    # for now assuming five actions
        self.current_state = None   # state agent is in
        self.observed_state = None  # state agent is in after taking action a
        self.last_action = None
        self.threshold = 0.15
        self.close = 0.10
        self.sensors_close = 0
        self.sensors_triggered = 0
        self.previous_sensors = 0
        self.distance = 0
        self.total_reward = []
        self.collision_list = []


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: observed front, right, left, front&right, front&left, nothing
        returns int
        """
        read = self.rob.read_irs()[3:]
        sensors = [np.inf if x == False else x for x in read]   #only take front sensors
        self.previous_sensors = self.sensors_triggered
        self.sensors_triggered = sum([x < self.threshold for x in sensors])
        self.sensors_close = sum([x < self.close for x in sensors])
        if all([x == np.inf for x in sensors]):
            return 0    # state 0 is no observations
        elif all([x < self.threshold for x in sensors]):
            return 1    # all sensors are triggered
        elif sensors[2] < self.threshold:
            if sensors[0] < self.threshold or sensors[1] < self.threshold:
                return 2    # object to the right and front
            elif np.argmin(sensors) == 3 or np.argmin(sensors) == 4:
                return 3    # object to the left and front
            else:
                return 4    # object to directly the front only
        elif sensors[3] < self.threshold or sensors[4] < self.threshold:
            return 5        # object directly to left
        elif sensors[0] < self.threshold or sensors[1] < self.threshold:
            return 6        # object to the right

        return 0

    def update_distance(self):
        self.distance = 0
        xdiff = abs(self.pos_before[0] - self.pos_after[0])
        ydiff = abs(self.pos_before[1] - self.pos_after[1])
        dist = math.sqrt(xdiff**2 + ydiff**2)
        if dist > 0.1:
            self.distance = dist
    
    def get_position(self):
        position = self.rob.getPosition()[1]
        return [position[0], position[1]]
            
    def action(self, state):
        """
        Use epsilon greedy policy to determine action
        runs certain action based on index, currently 5 actions
        """
        if np.random.binomial(1, self.eps) == 1:
            action_index = np.random.choice([0, 1, 2, 3, 4])  #pick a random action based on the index. see actions.py
        else:
            action_index = np.argmax(self.q_values[state])  # pick the best action based on q-values
        self.last_action = action_index
        select_action(self.rob, action_index)


    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        reward = 0
        reward = 50*self.distance - self.sensors_triggered + self.previous_sensors - self.sensors_close  #penalize trigering sensors and further so when closer to objects
        print(reward)
        return reward
    
    def action_eval(self, state):
        action_index = np.argmax(self.q_values[state])
        select_action(self.rob, action_index)



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
        
    def collision(self):
        read = self.rob.read_irs()[3:]
        col = 0.02
        sensors = [np.inf if x == False else x for x in read]   #only take front sensors
        if any([x < col for x in sensors]):
            return 1   # state 0 is no observations
        return 0


def evaluation(agent, evalsteps=100):
    agent.rob.play_simulation()
    time.sleep(3)
    agent.rob.move(10, -10, np.random.randint(1, 10) * 300)
    agent.current_state = agent.get_state()
    time.sleep(1)
    totalreward = 0
    totalsteps = 0
    collisioncount = 0
    agent.total_distance = 0
    for step in range(evalsteps):
        agent.pos_before = agent.get_position()
        agent.action_eval(agent.current_state)  # play best move according to policy
        time.sleep(0.5)
        agent.pos_after = agent.get_position()
        agent.update_distance()
        agent.observed_state = agent.get_state()
        collisioncount += agent.collision()
        time.sleep(0.5)
        reward = agent.get_reward()
        totalreward += reward
        agent.current_state = agent.observed_state
    agent.total_reward.append(totalreward)
    agent.collision_list.append(collisioncount)
    agent.rob.stop_world()
    time.sleep(1)

def plot_metrics(agent):
    print("Total reward: ", agent.total_reward)
    print("Num. of collisions: ", agent.collision_list)
    plt.plot(agent.total_reward)
    plt.show()
    plt.plot(agent.collision_list)
    plt.show()


def train_loop(rob, episodes=10, steps=50):
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
        for step in range(steps):
            agent.pos_before = agent.get_position()
            agent.action(agent.current_state)
            time.sleep(1)
            agent.observed_state = agent.get_state()
            agent.pos_after = agent.get_position()
            agent.update_distance()
            reward = agent.get_reward()
            agent.calc_Q_values(agent.last_action, reward)
            agent.current_state = agent.observed_state

        if agent.eps > agent.epsmin:
            agent.eps -= agent.decay

        for key, values in agent.q_values.items():
            print(f"State {key} Q-values: {values}")
        agent.rob.stop_world()
        time.sleep(1)
        evaluation(agent, 20)
    plot_metrics(agent)

    if os.path.exists("Qvalues.txt"):
        os.remove('Qvalues.txt')
    for key, values in agent.q_values.items():
        f = open('Qvalues.txt', 'a')
        string = ""
        for value in values:
            string = string + "," + str(value)
        f.write(string)
        f.close()

