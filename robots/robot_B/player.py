'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import eval7
import numpy as np
from math import ceil

from perm import *

# generate list of all hands to use for range calculations
SUITS = ['s', 'c', 'h', 'd']
CARDS = [v+s for v in ALL_RANKS for s in SUITS]
HANDS = [(a,b) for i,a in enumerate(CARDS) for b in CARDS[:i]]

# tweak these to adjust our preflop ranking of hands
# pair = 22, suited = 2 when we have perfect knowledge of card rankings,
# but with unknown rankings we may want to increase these?
PREFLOP_PAIR_VALUE = 22 # maybe value pairs more since card ranking is unknown?
PREFLOP_SUITED_VALUE = 2

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        # store preflop hand rankings and hands (sorted),
        # updating as new evidence comes in
        self.hand_values = [dict((hand, 0) for hand in HANDS) for order in order_ensemble]
        self.sorted_hands = [[] for order in order_ensemble]
        self.value_ranks_ensemble = []
        for order in order_ensemble:
            self.value_ranks_ensemble.append({c: order.index(c) for c in ALL_RANKS})
        #self.value_ranks = value_ranking(order_ensemble)
        self.evidence_updated = True

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        #big_blind = bool(active)  # True if you are the big blind
        print('New round #{}'.format(game_state.round_num))
        if self.evidence_updated:
            # update preflop hand rankings with new evidence
            for i,value_ranks in enumerate(self.value_ranks_ensemble):
                for hand in HANDS:
                    values = [value_ranks[c[0]] for c in hand]
                    self.hand_values[i][hand] = (2*max(values) + min(values)
                                                 + (PREFLOP_PAIR_VALUE if values[0] == values[1] else 0)
                                                 + (PREFLOP_SUITED_VALUE if hand[0][1] == hand[1][1] else 0))
                self.sorted_hands[i] = sorted(HANDS, key=lambda l: self.hand_values[i][l], reverse=True)
            self.evidence_updated = False
        self.sorted_street = [0 for order in order_ensemble]
        self.range = self.sorted_hands[:]

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        #previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        print('Game clock: {:.6f}s'.format(game_state.game_clock))
        evi = get_perm_evidence(terminal_state, active)
        if evi is not None:
            partial_order.add_evidence(evi)
            print('New partial order:', partial_order)
            print('Refreshing ensemble...')
            global order_ensemble
            order_ensemble = monte_carlo_perms(n=10) # refresh ensemble
            print('[...]')
            for order in order_ensemble[-10:]:
                print(order)
            self.value_ranks_ensemble = []
            for order in order_ensemble:
                self.value_ranks_ensemble.append({c: order.index(c) for c in ALL_RANKS})

            #self.value_ranks = value_ranking(order_ensemble)
            self.evidence_updated = True
            
 
    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        #street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, river, or turn respectively
        #my_cards = round_state.hands[active]  # your cards
        #board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        #continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        #if RaiseAction in legal_actions:
        #    min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
        #    min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
        #    max_cost = max_raise - my_pip  # the cost of a maximum bet/raise


        # "check" down if already all-in
        if len(legal_actions) == 1:
            return CheckAction()

        # x/f to victory if we can
        print(game_state.bankroll, NUM_ROUNDS, game_state.round_num)
        if game_state.bankroll > ceil((NUM_ROUNDS-game_state.round_num) * (SMALL_BLIND + BIG_BLIND)/2):
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # Simple hardcoded / eval7 strategy.
        street = round_state.street
        my_hand = round_state.hands[active]
        board = round_state.deck[:street]

        # reorder cards for consistent key in dict
        my_hand = my_hand[::-1] if CARDS.index(my_hand[0]) < CARDS.index(my_hand[1]) else my_hand

        # raise fraction / sizes
        # these are parameters to be tweaked (maybe by stats on opponent)
        if street == 0:
            RAISE_FRACTION = 0.33 # continue with 1/3 of our defends as 3bets
            RAISE_SIZE = 1 # raise (1x)pot
            MDF_FACTOR = 1.05 # slightly looser pre
        else:
            RAISE_FRACTION = 0.15
            RAISE_SIZE = 0.75 # bet 3/4 pot
            MDF_FACTOR = 0.6 # slightly tighter post


        if my_pip == 1 and street == 0: # open 85% of our button preflop
            raise_range = mdf = bluff_range = 0.85
        else:
            # calculate defend frequency and raise frequency
            if my_pip == 0 and opp_pip == 0:
                raise_range = RAISE_FRACTION
                mdf = RAISE_FRACTION
            else:
                mdf = MDF_FACTOR * 2*my_contribution / ((opp_pip - my_pip) + 2*my_contribution)
                raise_range = mdf*RAISE_FRACTION

            bluff_range = mdf*(1 + (RAISE_SIZE)/(1+2*RAISE_SIZE) * RAISE_FRACTION)

        bluff_range = mdf ### TEMP: other bots don't seem to fold too much so let's just not bluff for now. let's try changing this later

        # compute all three actions
        raise_size = min(int(opp_pip + RAISE_SIZE*2*opp_contribution), my_pip + my_stack)
        action_options = [
            RaiseAction(raise_size),
            CallAction(),
            CheckAction(),
            FoldAction()
        ]
        action_votes = [0, 0, 0, 0]
        
        for i,value_ranks in enumerate(self.value_ranks_ensemble):
            # re-sort range based on board texture
            if street > 0 and street != self.sorted_street[i]:
                hand_values = {}

                trans_board = [ALL_RANKS[value_ranks[c[0]]]+c[1] for c in board]
                new_range = []
                for hand in self.range[i]:
                    if not hand[0] in board and not hand[1] in board:
                        trans_hand = [ALL_RANKS[value_ranks[c[0]]]+c[1] for c in hand]
                        value = eval7.evaluate([eval7.Card(s) for s in trans_hand + trans_board])
                        hand_values[hand] = value
                        new_range += [hand]

                self.range[i] = sorted(new_range, key=lambda l: hand_values[l], reverse=True)
                self.sorted_street[i] = street

                # print('====================')
                # print('Ranks:', value_ranks)
                # print(my_hand, board)
                # print([ALL_RANKS[value_ranks[c[0]]]+c[1] for c in my_hand], [ALL_RANKS[value_ranks[c[0]]]+c[1] for c in board], '(translated)')
                # print('Top 20 hands in range:', [(h, hand_values[h]) for h in self.range[:20]])
                # print('Range: ', self.range)

            # rank current hand in our range
            N = len(self.range[i])
            hand_position = self.range[i].index(tuple(my_hand)) / N

            #print('Hand %:', hand_position, my_hand, [ALL_RANKS[value_ranks[c[0]]]+c[1] for c in my_hand], value_ranks)
            #print(raise_range, '(raise)', mdf, '(defend)', bluff_range, '(bluff)')

            # determine whether hand is a raise/call/bluff-raise/check/fold

            # currently commented out range updates; to be "unexploitable" our ranges should be
            # restricted at each decision (otherwise can be overbluffed if we show weakness), but
            # that results in us overvaluing weak hands.
            # e.g. if we check flop, we eliminate all ~pair+ holdings
            #      then on turn if we bet the top 33% of our range, we'll have to start value-betting high card hands
            # to do it properly we need some "slowplay" range to protect our delayed value hands
            if (hand_position < raise_range or mdf < hand_position < bluff_range) and opp_contribution < STARTING_STACK:
                if street == 0:
                    self.range[i] = self.range[i][:ceil(N*raise_range)] + self.range[i][int(N*mdf):int(ceil(N*bluff_range))]
                action_votes[0] += 1
            elif hand_position < mdf and opp_pip > my_pip:
                if street == 0:
                    self.range[i] = (self.range[i][:int(N*raise_range)] if opp_pip == my_pip+my_stack else []) + self.range[i][int(N*raise_range):int(ceil(N*mdf))]
                action_votes[1] += 1
            elif my_pip == opp_pip:
                if street == 0:
                    self.range[i] = self.range[i][int(N*raise_range):int(ceil(N*mdf))] + self.range[i][int(N*bluff_range):]
                action_votes[2] += 1
            else:
                action_votes[3] += 1
        action = action_options[np.argmax(action_votes)]
        print(f'Action votes raise {action_votes[0]} call {action_votes[1]} check {action_votes[2]} fold {action_votes[3]}')
        print(f'Action {action}')
        return action


if __name__ == '__main__':
    run_bot(Player(), parse_args())
