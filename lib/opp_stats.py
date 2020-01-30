class OpponentStats:
    def __init__(self):
        self.limp = -1
        self.fold = -1
        self.call = -1
        self.rais = -1
        self.tbet = -1

        self.preflop_rates = {
            'limp': [0,0],
            'fold': [0,0],
            'call': [0,0],
            'raise': [0,0],
            '3bet': [0,0]
        }
        self.last_street = 0
        self.last_action = None
        # TODO: other streets?

    def update_stats_full(self,terminal_state,active):
        #parse the streets, keep only the flops
        r = terminal_state.previous_state
        street_action = []
        while True:
            if r.street == 0:
                street_action.append([r.street,r.pips,active,r.button])
            
            r = r.previous_state
            if r == None:
                break

        street_action = street_action[::-1]
        opp = (active + 1)%2

        #they go first
        if opp == 0:
            if len(street_action) == 1:
                self.preflop_rates['fold'][0] += 1
                self.preflop_rates['fold'][1] += 1

                self.preflop_rates['limp'][0] += 0
                self.preflop_rates['limp'][1] += 1

                self.preflop_rates['raise'][0] += 0
                self.preflop_rates['raise'][1] += 1
            
            else:
                if street_action[1][1][opp] == 2:
                    self.preflop_rates['fold'][0] += 0
                    self.preflop_rates['fold'][1] += 1

                    self.preflop_rates['limp'][0] += 1
                    self.preflop_rates['limp'][1] += 1

                    self.preflop_rates['raise'][0] += 0
                    self.preflop_rates['raise'][1] += 1
                
                else:
                    self.preflop_rates['fold'][0] += 0
                    self.preflop_rates['fold'][1] += 1

                    self.preflop_rates['limp'][0] += 0
                    self.preflop_rates['limp'][1] += 1

                    self.preflop_rates['raise'][0] += 1
                    self.preflop_rates['raise'][1] += 1

        #we go first
        else:
            #if we fold pre, no info
            if len(street_action) == 1:
                return

            if len(street_action) == 2:
                pip = street_action[1]
                if pip[opp] == pip[active]:
                    self.preflop_rates['call'][0] += 1
                    self.preflop_rates['call'][1] += 1

                    self.preflop_rates['fold'][0] += 0
                    self.preflop_rates['fold'][1] += 1

                    self.preflop_rates['3bet'][0] += 0
                    self.preflop_rates['3bet'][1] += 1

                else:
                    self.preflop_rates['call'][0] += 0
                    self.preflop_rates['call'][1] += 1

                    self.preflop_rates['fold'][0] += 1
                    self.preflop_rates['fold'][1] += 1

                    self.preflop_rates['3bet'][0] += 0
                    self.preflop_rates['3bet'][1] += 1

            #they raised us up, more streets of action
            else:
                self.preflop_rates['3bet'][0] += 1
                self.preflop_rates['3bet'][1] += 1

                self.preflop_rates['call'][0] += 0
                self.preflop_rates['call'][1] += 1

                self.preflop_rates['fold'][0] += 0
                self.preflop_rates['fold'][1] += 1


    if self.preflop_rates['flop'][1] >= 10:
        self.flop = self.preflop_rates['flop'][0]/self.preflop_rates['flop'][1]
    
    if self.preflop_rates['limp'][1] >= 10:
        self.limp = self.preflop_rates['limp'][0]/self.preflop_rates['limp'][1]
    
    if self.preflop_rates['call'][1] >= 10:
        self.call = self.preflop_rates['call'][0]/self.preflop_rates['call'][1]
    
    if self.preflop_rates['raise'][1] >= 10:
        self.rais = self.preflop_rates['raise'][0]/self.preflop_rates['raise'][1]
    
    if self.preflop_rates['3bet'][1] >= 10:
        self.tbet = self.preflop_rates['3bet'][0]/self.preflop_rates['3bet'][1]
