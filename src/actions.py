def forward(rob):
    rob.move(20, 20, 2000)


def turn_left(rob):
    rob.move(-20, 20, 1000)


def turn_right(rob):
    rob.move(20, -20, 1000)


def select_action(rob, index):
    if index == 0:
        forward(rob)
    elif index == 1:
        turn_left(rob)
    elif index == 2:
        turn_right(rob)
