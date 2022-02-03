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
        self.eps = 0.40
        self.epsmin = 0.05
        self.decay = 0.04
        self.alpha = 0.3
        self.alphamin = 0.01
        self.rob = rob
        self.last_action = None
        self.q_values = {}
        self.max_reward = 0
        self.current_state = None   # state agent is in
        self.observed_state = None  # state agent is in after taking action a
        self.terminal_state = False
        self.rewards = 0
        self.cum_reward = []
        self.steps = []
        self.total_reward = []
        self.steps_col = []
        self.total_lost = []
        self.col_threshold = 0.06
        self.threshold = 0.15
        self.green_low_bound = np.array([0, 100, 0])
        self.green_upper_bound = np.array([40, 255, 40])
        self.red_low_bound = np.array([0, 0, 100])
        self.red_upper_bound = np.array([60, 60, 255])
        self.width = None
        self.height = None
        self.food_eaten = 0
        self.last_position = None
        self.counter = 0
        self.collect = False
        self.prev_collect = False
        self.initial_pickup = True
        self.col = 0
        self.cols =["red","green"]
        self.last_red = None

    def read_q_values(self, states, numactions):
        file = open("Qvalues_max_perfect.txt", 'r')
        string = file.read()
        listvals = string.split(",")[1:]
        listvals = [float(x) for x in listvals]
        for i, state in enumerate(states):
            if i == 0:
                self.q_values[state] = listvals[0: numactions]
            else:
                self.q_values[state] = listvals[i*numactions : i * numactions + numactions]
        print(self.q_values)


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
        
    def get_xy(self,mask):
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


    def get_blob_location(self):
        image = self.rob.get_image_front()
        self.width = image.shape[0]
        self.height = image.shape[1]
        cv2.imwrite("test.png", image)
        #print(image.shape)
        mask = cv2.inRange(image, self.red_low_bound, self.red_upper_bound)
        mask2 = cv2.inRange(image, self.green_low_bound, self.green_upper_bound)
        cv2.imwrite("testr.png", mask)
        cv2.imwrite("testg.png", mask2)
        rx, ry = self.get_xy(mask)
        gx, gy = self.get_xy(mask2)
        return rx, ry, gx, gy


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: No object, object to left, object in middle, object to right, object touched
        returns int
        """
        read = self.rob.read_irs()[3:]
        sensors = [np.inf if x == False else x for x in read]
        front = sensors[2]
        other = [True if x < self.threshold else False for x in np.delete(sensors, 2)]
   
        rx, ry, gx, gy = self.get_blob_location()
        red = 0
        green = 0
        temp = 0
        
        if (rx is None):    
            red = 0            # red object not detected
        elif ry < self.height/2:
            red = 1            # red object detected far
        elif rx  < (self.width/3):
            red = 2            # red object detected left
        elif rx  >= (self.width/3) and rx <= (self.width/3 * 2):
            red = 3            # red object detected center
        elif rx  > (self.width/3 * 2):
            red = 4            # red object detected right
            
        if (gx is None):    
            green = 5            # green object not detected
        elif gy < self.height/2:
            green = 6            # green object detected far
        elif gx  < (self.width/3):
            green = 7            # green object detected left
        elif gx  >= (self.width/3) and gx <= (self.width/3 * 2):
            green = 8            # green object detected center
        elif gx  > (self.width/3*2):
            green = 9            # green object detected right
        
        if front < self.col_threshold and sum(other) < 1:
            self.collect = True                     # Assume collected when only detected very close in front sensor
        elif front > self.col_threshold or not (rx is None):
            self.collect = False                    # Assume lost when see Red again or front sensor not close
        self.prev_collect = self.collect
        if self.collect:
            temp = 10
        else:
            temp = 0
            
        
        if self.col < 1:
            return red + temp
        else:
            return green + temp               

        return 0


    def action(self, state):
        """
        Use epsilon greedy policy to determine action
        runs certain action based on index, currently 5 actions
        """


    def action_eval(self, state):
        action_index = np.argmax(self.q_values[state])
        if action_index < 3:
            select_action(self.rob, action_index)
        elif self.col < 1:
            self.col = 1
        else:
            self.col = 0



    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        ## Give high reward when food is delivered
        
        if self.rob.base_detects_food():
            self.terminal_state = True
            return 150
            
        ## food captured, sensor state
        ## high reward
        
        ## Terminate and give high penalty when stuck
        if self.counter > 50:
            self.terminal_state = True
            return -20
        
        ## Give high reward when food is initially collected
        elif self.collect:
            if self.initial_pickup:
                self.initial_pickup = False
                return 150

        return -1


    def calc_Q_values(self, action, reward):
        """
        update q_values
        """
        state = self.current_state
        _state = self.observed_state

        print("Current state is ", state, " obsereved state is ", _state)
        current_q = self.q_values[state][action]
        next_q = np.max(self.q_values[_state])
        a = self.alpha
        g = self.gamma
        self.q_values[state][action] = current_q + a*(reward + g*next_q - current_q)




def evaluation(agent, evalsteps=100):
    agent.collect = False
    agent.initial_pickup = True
    agent.counter = 0
    agent.rob.play_simulation()
    time.sleep(3)
    #agent.rob.move(10, -10, np.random.randint(1, 10) * 300) # random orientation
    agent.current_state = agent.get_state()
    time.sleep(1)
    colsteps = 0
    total_reward = 0
    totalsteps = 0
    lost = 0
    for step in range(evalsteps):
        if agent.terminal_state:
        
            agent.terminal_state = False
            agent.rob.move(0,0,100)
            break
        agent.action_eval(agent.current_state)  # play best move according to policy
        time.sleep(0.2)
        agent.observed_state = agent.get_state()
        time.sleep(0.2)
        if agent.current_state == 0 or agent.current_state == 5 or agent.current_state == 10 or agent.current_state == 15: # Seeing noting (before and after collected)
            if agent.observed_state == 0 or agent.current_state == 5 or agent.current_state == 10 or agent.current_state == 15:
                agent.counter +=1
            else:
                agent.counter = 0
        reward = agent.get_reward()        
        total_reward += reward
        agent.current_state = agent.observed_state
        print("Current step: ", step)
        state = agent.current_state
        _state = agent.observed_state

        print("Current state is ", state, " obsereved state is ", _state)
        totalsteps += 1
        if agent.counter > 50:
            totalsteps = 0
        if agent.initial_pickup:
            colsteps = totalsteps
        if not agent.collect:
            if agent.prev_collect:
                lost += 1
    agent.total_lost.append(lost)
    agent.total_reward.append(total_reward)
    agent.steps_col.append(colsteps)
    agent.steps.append(totalsteps)
    agent.rob.move(0,0,100)
    agent.rob.stop_world()
    time.sleep(1)


def plot_metrics(agent):
    print("Total reward: ", agent.total_reward)
    print("Steps untill collected: ", agent.steps_col)
    print("Steps untill end: ", agent.steps)
    print("Training: ", agent.rewards)
    plt.plot(agent.total_reward)
    plt.title("Total reward", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.savefig("Reward.png")
    plt.clf()
    plt.plot(agent.steps_col)
    plt.title("Number of steps until pick-up", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Steps", fontsize=16)
    plt.savefig("Steps_col.png")
    plt.clf()
    plt.plot(agent.steps)
    plt.title("Number of steps until brought to base", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Steps", fontsize=16)
    plt.savefig("Steps_fin.png")
    plt.clf()
    plt.plot(agent.cum_reward)
    plt.title("Cumulative reward - training", fontsize=16)
    plt.xlabel("Steps", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.savefig("Cum_reward.png")
    plt.clf()
    plt.plot(agent.total_lost)
    plt.title("Times food was dropped", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Times dropped", fontsize=16)
    plt.savefig("lost.png")
    plt.clf()


def train_loop(rob, episodes=100, steps=1000, evaluations=5):
    """
    Combines all of the above to run a training loop and update the Q-values
    Does 15 training epochs with 50 steps per epoch
    returns nothing, should likely store values of self.q_values in file
    """
    agent = Agent(rob)
    agent.read_q_values(range(20),4)
    for ev in range(evaluations):
        print("Current eval round: ", ev)
        evaluation(agent, 200)
        
    plot_metrics(agent)