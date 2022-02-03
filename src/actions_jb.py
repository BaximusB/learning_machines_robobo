def forward(rob):
    rob.move(10, 10, 2000)

def turn_left(rob):
    rob.move(-5, 5, 500)

def turn_right(rob):
    rob.move(5, -5, 500)

def turn_left_long(rob):
    rob.move(-5, 5, 1500)

def turn_right_long(rob):
    rob.move(5, -5, 1500)

def turn_left_v_long(rob):
    rob.move(-5, 5, 4000)
    
def turn_right_v_long(rob):
    rob.move(5, -5, 4000)


def select_action(rob, index, col):
    if index == 0:
        print('Moving forward')
        forward(rob)
        return -1
    elif index == 1:
        print('Turning left')
        turn_left(rob)
        return -1
    elif index == 2:
        print('Turning right')
        turn_right(rob)
        return -1
    elif index == 3:
        print('Switch mask')
        # mask switch
        if col == 0:            
            return 1
        return 0
        
def select_action_old(rob, index):
    if index == 0:
        print('Moving forward')
        forward(rob)
    elif index == 1:
        print('Turning left')
        turn_left(rob)
    elif index == 2:
        print('Turning right')
        turn_right(rob)
    elif index == 3:
        print('Turning long left')
        turn_left_long(rob)
    elif index == 4:
        print('Turning long right')
        turn_right_long(rob)
    elif index == 5:
        print('Turning very long left')
        turn_left_v_long(rob)
    elif index == 6:
        print('Turning very long right')
        turn_right_v_long(rob)