class OpponentStats:
    def __init__(self):
        self.preflop_rates = {
            'fold': [0,0],
            'call': [0,0],
            'raise': [0,0]
        }
        self.last_street = 0
        # TODO: other streets?
    def reset_round(self):
        self.last_street = 0
    def update_stats(self, round_state):
        street = round_state.street
        opponent_action = round_state.last_action
        if street == 0: # preflop
            pass
        else:
            # TODO
            pass
