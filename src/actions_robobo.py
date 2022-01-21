def forward(rob):
    rob.move(10, 10, 300)

def turn_left(rob):
    rob.move(-10, 10, 200)

def turn_right(rob):
    rob.move(10, -10, 350)

def select_action(rob, index):
    if index == 0:
        forward(rob)
    elif index == 1:
        turn_left(rob)
    elif index == 2:
        turn_right(rob)

