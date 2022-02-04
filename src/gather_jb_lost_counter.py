import numpy as np
from actions_jb import *
import time
import os
import matplotlib.pyplot as plt
import cv2

# define agent
class Agent:
    def __init__(self, rob):
        self.gamma = 0.9
        self.eps = 0.4
        self.epsmin = 0.05
        self.decay = 0.04
        self.alpha = 0.3
        self.rob = rob
        self.last_action = None
        self.q_values = {x: [np.random.uniform(0, 1) for _ in range(4)] for x in range(20)}
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
        self.counter = 0
        self.collect = False
        self.prev_collect = False
        self.initial_pickup = True
        self.col = 0
        self.cols =["red","green"]

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
        mask = cv2.inRange(image, self.red_low_bound, self.red_upper_bound)
        mask2 = cv2.inRange(image, self.green_low_bound, self.green_upper_bound)
        cv2.imwrite("testr.png", mask)
        cv2.imwrite("testg.png", mask2)
        rx, ry = self.get_xy(mask)
        gx, gy = self.get_xy(mask2)
        return rx, ry, gx, gy

    def get_state(self):
        # get sensor readings
        read = self.rob.read_irs()[3:]
        sensors = [np.inf if x == False else x for x in read]
        front = sensors[2]
        other = [True if x < self.threshold else False for x in np.delete(sensors, 2)]
    
        # get blob readings
        rx, ry, gx, gy = self.get_blob_location()
        
        # detect if food is caught by checking if only front sensor is triggered
        if front < self.col_threshold and sum(other) < 1:
            self.collect = True                     # Assume collected when only detected very close in front sensor
        elif front > self.col_threshold or not (rx is None):
            self.collect = False                    # Assume lost when see Red again or front sensor not close
        self.prev_collect = self.collect
        
        # collected states
        if self.collect:
            if self.col == 0:
                if (rx is None):    
                    return 0            # red object not detected
                elif ry < self.height/2:
                    return 1            # red object detected far
                elif rx  < (self.width/5 * 2):
                    return 2            # red object detected left
                elif rx  >= (self.width/5 * 2) and rx <= (self.width/5 * 3):
                    return 3            # red object detected center
                elif rx  > (self.width/5 * 3):
                    return 4            # red object detected right
            
            if self.col == 1: 
                if (gx is None):    
                    return 5            # green object not detected
                elif gy < self.height/2:
                    return 6            # green object detected far
                elif gx  < (self.width/5 * 2):
                    return 7            # green object detected left
                elif gx  >= (self.width/5 * 2) and gx <= (self.width/5 * 3):
                    return 8            # green object detected center
                elif gx  > (self.width/5 * 3):
                    return 9            # green object detected right

        # non-collected states
        if not self.collect:
            if self.col == 0:
                if (rx is None):    
                    return 10            # red object not detected
                elif ry < self.height/2:
                    return 11           # red object detected far
                elif rx  < (self.width/5 * 2):
                    return 12            # red object detected left
                elif rx  >= (self.width/5 * 2) and rx <= (self.width/5 * 3):
                    return 13            # red object detected center
                elif rx  > (self.width/5 * 3):
                    return 14            # red object detected right
            
            if self.col == 1:
                if (gx is None):    
                    return 15            # green object not detected
                elif gy < self.height/2:
                    return 16            # green object detected far
                elif gx  < (self.width/5 * 2):
                    return 17            # green object detected left
                elif gx  >= (self.width/5 * 2) and gx <= (self.width/5 * 3):
                    return 18            # green object detected center
                elif gx  > (self.width/5 * 3):
                    return 19            # green object detected right

    def action(self, state):
        action_index = np.argmax(self.q_values[state])
        if np.random.binomial(1, self.eps) == 1:
            print('Exploring')
            action_index = np.random.choice([x for x in range(4)])
                
        self.last_action = action_index
        a = select_action(self.rob, action_index, self.col)
        
        # mask switch
        if a == 1:
            self.col = 1
        elif a == 0:
            self.col = 0
    
    def action_eval(self, state):
        action_index = np.argmax(self.q_values[state])
        select_action(self.rob, action_index)

    def get_reward(self):
        ## Give high reward when food is delivered        
        if self.rob.base_detects_food():
            self.terminal_state = True
            return 50
        
        # if in collected state and we see green base in centre before and after action taken
        if self.current_state == 8:
            if self.observed_state == 8:
                return 0.5
            
        # if not collected state, and we see red food in centre before and after action taken 
        if self.current_state == 13:
            if self.observed_state == 13:
                return 0.5
        
        ## Terminate and give high penalty when stuck without food
        if self.counter > 50 and not self.collect:            
            self.terminal_state = True
            return -50
        
        ## Terminate and give mild penalty when stuck with food
        if self.counter > 100 and self.collect:            
            self.terminal_state = True
            return -25
        
        ## Give small reward when food is initially collected
        elif self.collect:
            if self.initial_pickup:
                self.initial_pickup = False
                return 30
        
        # increase penalty the more food is lost
        return -1 * (self.lost/2 + 1)

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
    # set params
    agent.collect = False
    agent.initial_pickup = True
    agent.counter = 0
    agent.lost = 0
    
    # start simulation
    agent.rob.play_simulation()
    time.sleep(3)
    
    # get state
    agent.current_state = agent.get_state()    
    time.sleep(1)
    
    colsteps = 0
    total_reward = 0
    totalsteps = 0
    
    # evaluate
    for step in range(evalsteps):
        if agent.terminal_state:
            agent.terminal_state = False
            agent.rob.move(0,0,100)
            break
        
        # play best move according to policy
        agent.action_eval(agent.current_state)
        time.sleep(0.2)
        agent.observed_state = agent.get_state()
        time.sleep(0.2)
        
        # Seeing nothing (before and after collected) - handle being stuck
        if agent.current_state == 0 or agent.current_state == 5 or agent.current_state == 10 or agent.current_state == 15:
            if agent.observed_state == 0 or agent.observed_state == 5 or agent.observed_state == 10 or agent.observed_state == 15:
                agent.counter +=1
            else:
                agent.counter = 0
        else:
            agent.counter = 0
        
        # get reward
        reward = agent.get_reward()        
        total_reward += reward
        
        # update current state
        agent.current_state = agent.observed_state
        
        # log
        print("Current step: ", step)
        
        totalsteps += 1
        if agent.initial_pickup:
            colsteps = totalsteps
        
        # check if robot lost food
        if not agent.collect:
            if agent.prev_collect:
                agent.lost += 1
    
    # update result lists
    agent.total_lost.append(agent.lost)
    agent.total_reward.append(total_reward)
    agent.steps_col.append(colsteps)
    agent.steps.append(totalsteps)
    
    # stop simulation
    agent.rob.move(0,0,100)
    agent.rob.stop_world()
    time.sleep(1)

def plot_metrics(agent):
    print("Total reward: ", agent.total_reward)
    print("Steps until collected: ", agent.steps_col)
    print("Steps until end: ", agent.steps)
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
    # init agent
    agent = Agent(rob)
    
    # train
    for episode in range(episodes):
        # set params
        agent.collect = False
        agent.initial_pickup = True
        agent.counter = 0
        agent.lost = 0
        
        # start simulation
        agent.rob.play_simulation()
        time.sleep(2)
        rob.set_phone_tilt(26, 10)
        
        # get state
        agent.current_state = agent.get_state()

        # train
        for step in range(steps):
            if agent.terminal_state:
                agent.terminal_state = False
                agent.rob.move(0,0,100)
                break
            
            # do action
            agent.action(agent.current_state)
            time.sleep(0.2)
            
            # observe state
            agent.observed_state = agent.get_state()            
            time.sleep(0.2)
            
            # log
            print("Current episode: ", episode, ", current step: ", step)
            print(f"Current state: {agent.current_state}, took action {agent.last_action}")            
            print("Collected food: ", agent.collect)
            print("Current mask: ", agent.cols[agent.col])
            
            # Seeing nothing (before and after collected) - handle being stuck
            if agent.current_state == 0 or agent.current_state == 5 or agent.current_state == 10 or agent.current_state == 15:
                if agent.observed_state == 0 or agent.observed_state == 5 or agent.observed_state == 10 or agent.observed_state == 15:
                    agent.counter +=1
                else:
                    agent.counter = 0
            else:
                agent.counter = 0
            
            # check if robot lost food   
            if not agent.collect:
                if agent.prev_collect:
                    agent.lost += 1
            
            # get reward
            reward = agent.get_reward()            
            agent.rewards += reward
            agent.cum_reward.append(agent.rewards)
            agent.calc_Q_values(agent.last_action, reward)
            
            # update current state
            agent.current_state = agent.observed_state
            
            # log
            print('Collected reward: ' + str(reward))

        # decay exploration
        if agent.eps > agent.epsmin:
            agent.eps -= agent.decay
        if agent.eps < agent.epsmin:
            agent.eps = agent.epsmin

        # q-values
        for key, values in agent.q_values.items():
            print(f"State {key} Q-values: {values}")
            
        # stop simulation
        agent.rob.move(0,0,100)
        agent.rob.stop_world()
        time.sleep(1)
    
    # eval
    for ev in range(evaluations):
        print("Current eval round: ", ev)
        evaluation(agent, 150)
    
    # plot
    plot_metrics(agent)

    # update q-values
    if os.path.exists("Qvalues.txt"):
        os.remove('Qvalues.txt')
    for key, values in agent.q_values.items():
        f = open('Qvalues.txt', 'a')
        string = ""
        for value in values:
            string = string + "," + str(value)
        f.write(string)
        f.close()