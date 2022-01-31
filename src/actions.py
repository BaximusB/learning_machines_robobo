def forward(rob):
    rob.move(10, 10, 2000)

def turn_left(rob):
    rob.move(-10, 10, 250)

def turn_right(rob):
    rob.move(10, -10, 250)

def turn_left_long(rob):
    rob.move(-5, 5, 1500)

def turn_right_long(rob):
    rob.move(5, -5, 1500)

def select_action(rob, index):
    if index == 0:
        forward(rob)
    elif index == 1:
        turn_left(rob)
    elif index == 2:
        turn_right(rob)
    elif index == 3:
        turn_left_long(rob)
    elif index == 4:
        turn_right_long(rob)