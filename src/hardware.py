import numpy as np


from actions import *

class Robobo:
    def __init__(self, rob):
        self.threshold = 1000
        self.rob = rob
        self.q_vals = {}

    def read_q_values(self, states, numactions):
        file = open("./src/Qvalues.txt", 'r')
        string = file.read()
        listvals = string.split(",")[1:]
        for i, state in enumerate(states):
            if i == 0:
                self.q_vals[state] = listvals[0: numactions]
            else:
                self.q_vals[state] = listvals[i*numactions : i * numactions + numactions]


    def get_states(self):
        sensors = self.rob.read_irs()[3:]
        sensors -= 60      # make all values negative if there is noise
        if np.argmax(sensors) == 0 or np.argmax(sensors) == 1:
            if sensors[0] < self.threshold or sensors[1] < self.threshold:
                return 1  # object to the right

        elif np.argmax(sensors) == 2:
            if sensors[2] < self.threshold:
                return 2  # object directly in front

        elif np.argmin(sensors) == 3 or np.argmin(sensors) == 4:
            if sensors[3] < self.threshold or sensors[4] < self.threshold:
                return 3  # object to the left
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
        robobo.action(state)