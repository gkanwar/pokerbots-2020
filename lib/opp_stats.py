class OpponentStats:
    def __init__(self):
        self.preflop_rates = {
            'fold': [0,0],
            'call': [0,0],
            'raise': [0,0]
        }
        self.last_street = 0
        self.last_action = None
        # TODO: other streets?
    def reset_round(self):
        self.last_street = 0
    def update_stats(self, round_state):
        street = round_state.street
        opponent_action = round_state.last_action
        if street != self.last_street:
            if street == 3: # just flopped
                if isinstance(self.last_action, RaiseAction):
                    self.preflop_rates['raise'][0] += 1
                elif isinstance(self.last_action, CallAction) or isinstance(self.last_action, CheckAction):
                    self.preflop_rates['call'][0] += 1
                else:
                    raise RuntimeError('Should not reach here with fold')
                for k in self.preflop_rates:
                    self.preflop_rates[k][1] += 1
            else: # TODO: other streets
                pass
        self.last_street = street
        self.last_action = opponent_action
