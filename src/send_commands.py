#!/usr/bin/env python3

from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey

# from agent import *
import foraging
from hardware import *


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="")
    rob = robobo.SimulationRobobo().connect(address='192.168.1.101', port=19997)
    rob.set_phone_tilt(26, 10)
    foraging.train_loop(rob)

    # rob.play_simulation()
    # time.sleep(1)

    # im = rob.get_image_front()


    # test_robobo(rob, [0, 1, 2, 3], 3) # run to test robobo hardware

    # train_loop(rob) # run to train robobo

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    # cv2.imwrite("test_pictures.png",image)

    # time.sleep(0.1)

    # IR reading
    # for i in range(100):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # pause the simulation and read the collected food
    # rob.pause_simulation()
    #
    # # Stopping the simualtion resets the environment
    # rob.stop_world()


if __name__ == "__main__":
    main()
