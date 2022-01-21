import numpy as np


from actions import *

import time

class Robobo:
    def __init__(self, rob):
        self.threshold = 100
        self.rob = rob
        self.q_vals = {}
        self.noise = [9.0, 25.0, 8.0, 22.0, 28.0, 11.0, 38.0, 12.0]

    def read_q_values(self, states, numactions):
        file = open("./src/Qvalues.txt", 'r')
        string = file.read()
        listvals = string.split(",")[1:]
        listvals = [float(x) for x in listvals]
        for i, state in enumerate(states):
            if i == 0:
                self.q_vals[state] = listvals[0: numactions]
            else:
                self.q_vals[state] = listvals[i*numactions : i * numactions + numactions]
        print(self.q_vals)

    def get_states(self):
        sensors = self.rob.read_irs()[3:]
        sensors = [x - n for x in sensors for n in self.noise[3:]]  # remove noise from sensors

        if np.argmax(sensors) == 0 or np.argmax(sensors) == 1:
            if sensors[0] > self.threshold or sensors[1] > self.threshold:
                return 1  # object to the right
            return 0
        elif np.argmax(sensors) == 2:
            if sensors[2] > self.threshold:
                return 2  # object directly in front
            return 0
        elif np.argmax(sensors) == 3 or np.argmax(sensors) == 4:
            if sensors[3] > self.threshold or sensors[4] > self.threshold:
                return 3  # object to the left
            return 0
        else:
            return 0


    def action(self, state):
        action_index = np.argmax(self.q_vals[state])
        select_action(self.rob, action_index)


def test_robobo(rob, states, numactions):
    robobo = Robobo(rob)
    robobo.read_q_values(states, numactions)


    while True:
        state = robobo.get_states()
        time.sleep(0.5)
        robobo.action(state)
        time.sleep(0.5)