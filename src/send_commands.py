#!/usr/bin/env python3

from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey

from agent import *

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    # rob.play_simulation()

    train_loop(rob, episodes=10)

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
