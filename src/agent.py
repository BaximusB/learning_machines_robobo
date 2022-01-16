class Agent:
    def __init__(self, rob):
        """
        Each state has multiple Q-values, therefore q_values is dict() object with keys=state and values=list of q-vals
        num_states likely equals 6: 5 observing states and one state when nothing is observed
        """
        self.gamma = None
        self.eps = None
        self.alpha = None
        self.rob = rob
        self.q_values = {x: list() for x in range(6)}


    def get_state(self):
        """
        Should use infrared data to determine state.
        States: observed front, right, left, front&right, front&left, nothing
        returns int
        """
        pass


    def action(self):
        """
        Use epsilon greedy policy to determine action
        Easy with np.random.binomial and np.random.choice
        returns action or action_index, both int
        """
        pass


    def get_reward(self):
        """
        Negative if observed (could be -1, -5, -10)
        Positive if not observed (could be 1, 5, 10)
        returns int
        """
        pass


    def calc_Q_values(self):
        """
        Use formula to update self.q_values
        returns nothing
        """
        pass


    def train_loop(self, episodes=15, steps=100):
        """
        Combines all of the above to run a training loop and update the Q-values
        Does 15 training epochs with 100 steps per epoch
        returns nothing, should likely store values of self.q_values in file
        """
        pass