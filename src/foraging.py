import numpy as np
from actions import *
import time
import os
import math
import matplotlib.pyplot as plt
import cv2

class Agent:
    def __init__(self, rob):
        """
        Each state has multiple Q-values, therefore q_values is dict() object with keys=state and values=list of q-vals
        num_states likely equals 6: 5 observing states and one state when nothing is observed
        """
        self.gamma = 0.9
        self.eps = 0.34
        self.epsmin = 0.02
        self.decay = 0.04
        self.alpha = 0.4
        self.rob = rob
        self.last_action = None
        self.q_values = {x: [0, 0, 0, 0, 0] for x in range(5)}    # for now assuming five actions
        self.current_state = None   # state agent is in
        self.observed_state = None  # state agent is in after taking action a
        self.terminal_state = False
        self.rewards = 0
        self.steps = []
        self.total_reward = []
        self.food_list = []
        self.low_bound = np.array([0, 100, 0])
        self.upper_bound = np.array([40, 255, 40])
        self.width = None
        self.height = None
        self.food_eaten = 0
        self.last_position = None




    def get_closest(self, contours):
        if len(contours) == 1:
            return 0
        else:
            yvals = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    yvals.append(-10)
                    continue
                yvals.append(np.max(contour[:, :, 1]))
            return np.argmax(yvals)


    def get_blob_location(self):
        image = self.rob.get_image_front()
        self.width = image.shape[0]
        self.height = image.shape[1]
        cv2.imwrite("test.png", image)
        #print(image.shape)
        mask = cv2.inRange(image, self.low_bound, self.upper_bound)
        cv2.imwrite("test1.png", mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None
        contour_index = self.get_closest(contours)
        contour = contours[contour_index]
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, None
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        return x, y


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: No object, object to left, object in middle, object to right, object touched
        returns int
        """
        read = [np.inf if x is False else x for x in self.rob.read_irs()]

        x, y = self.get_blob_location()
        if (x is None):   # No object
            #if self.current_state == 1:
            #    return 4
            #if self.current_state == 3:
            #    return 5
            return 0

        if (x is None):   # No object
            return 0
        if y > self.height/2:
            return 4        # object is far
        if x < self.width/3:
            return 1       # object on left side
        if (x >= self.width/3) and (x <= (self.width/3 * 2)):
            # if any(read[x] < 0.05 for x in range(4, 7)):    # only look at front 3 sensors
            #     return 4    # object touched
            return 2        #object in middle
        if x > (self.width/3 * 2):

            #self.last_position = "R"

            return 3        # object on right side
        return 0


    def action(self, state):
        """
        Use epsilon greedy policy to determine action
        runs certain action based on index, currently 5 actions
        """
        action_index = np.argmax(self.q_values[state])
        if np.random.binomial(1, self.eps) == 1:
            action_index = np.random.choice([x for x in range(5)])
        self.last_action = action_index
        select_action(self.rob, action_index)


    def action_eval(self, state):
        action_index = np.argmax(self.q_values[state])
        select_action(self.rob, action_index)


    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        temp_food = self.food_eaten
        self.food_eaten = self.rob.collected_food()
        print("Collected food: ", self.food_eaten)
        if self.rob.collected_food() == 7:
            self.terminal_state = True
            #self.last_position = None
            return 20
        if (self.food_eaten - temp_food) > 0:
            #self.last_position = None
            return 20
        
        else:
            return 0

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




def evaluation(agent, evalsteps=50):
    agent.rob.play_simulation()
    time.sleep(3)
    agent.rob.move(10, -10, np.random.randint(1, 10) * 300) # random orientation
    agent.current_state = agent.get_state()
    time.sleep(1)
    totalsteps = 0
    for step in range(evalsteps):
        if agent.terminal_state:
            agent.terminal_state = False
            break
        agent.action_eval(agent.current_state)  # play best move according to policy
        time.sleep(0.2)
        agent.observed_state = agent.get_state()
        time.sleep(0.2)
        reward = agent.get_reward()
        agent.rewards += reward
        agent.current_state = agent.observed_state
        totalsteps += 1
    agent.total_reward.append(agent.rewards)
    agent.steps.append(totalsteps)
    agent.food_list.append(agent.food_eaten)
    agent.rob.stop_world()
    time.sleep(1)


def plot_metrics(agent):
    print("Total reward: ", agent.total_reward)
    print("Steps: ", agent.steps)
    print("Num. of food: ", agent.food_list)
    plt.plot(agent.total_reward)
    plt.title("Cumulative reward", fontsize=16)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.savefig("Cum_Reward.png")
    plt.clf()
    plt.plot(agent.steps)
    plt.title("Number of steps", fontsize=16)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Steps", fontsize=16)
    plt.savefig("Steps.png")
    plt.clf()
    plt.plot(agent.food_list)
    plt.title("Number of food gathered", fontsize=16)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Food", fontsize=16)
    plt.savefig("Food.png")
    plt.clf()


def train_loop(rob, episodes=20, steps=1000):
    """
    Combines all of the above to run a training loop and update the Q-values
    Does 15 training epochs with 50 steps per epoch
    returns nothing, should likely store values of self.q_values in file
    """
    agent = Agent(rob)
    for episode in range(episodes):
        agent.rob.play_simulation()
        time.sleep(2)
        _time = np.random.randint(1, 10)*300
        agent.rob.move(10, -10, _time)      # random-ish orientation
        time.sleep(1)
        agent.current_state = agent.get_state()

        for step in range(steps):
            if agent.terminal_state:
                agent.terminal_state = False
                break

            agent.action(agent.current_state)
            time.sleep(0.2)
            print("Current episode, step: ", episode)
            print(f"Current state: {agent.current_state}, took action {agent.last_action}")
            agent.observed_state = agent.get_state()
            time.sleep(0.2)
            reward = agent.get_reward()
            agent.calc_Q_values(agent.last_action, reward)
            agent.current_state = agent.observed_state

        if agent.eps > agent.epsmin:
            agent.eps -= agent.decay

        for key, values in agent.q_values.items():
            print(f"State {key} Q-values: {values}")
        agent.rob.stop_world()
        time.sleep(1)
        evaluation(agent, 100)
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