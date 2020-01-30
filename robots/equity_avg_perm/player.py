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
ALL_HANDS = [(a,b) for a in CARDS for b in CARDS]
def convert_e7(cards):
    return [eval7.Card(s) for s in cards]
HANDS_E7 = {hand: convert_e7(hand) for hand in ALL_HANDS}

# tweak these to adjust our preflop ranking of hands
# pair = 22, suited = 2 when we have perfect knowledge of card rankings,
# but with unknown rankings we may want to increase these?
PREFLOP_PAIR_VALUE = 22 # maybe value pairs more since card ranking is unknown?
PREFLOP_SUITED_VALUE = 2

def translate_hands(value_ranks, hands):
    trans_hands = []
    for hand in hands:
        trans_hands.append(tuple([ALL_RANKS[value_ranks[c[0]]]+c[1] for c in hand]))
    return trans_hands

class Player(Bot):
    '''
    A pokerbot.
    '''
    N_MCMC = 1000

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
        self.hand_values = dict((hand, 0) for hand in HANDS)
        self.sorted_hands = []
        self.value_ranks = value_ranking(order_ensemble)
        self.trans_hands = translate_hands(self.value_ranks, HANDS)
        self.evidence_updated = True

        self.random_hand = eval7.HandRange(','.join([ALL_RANKS[i]+ALL_RANKS[j] for i in range(len(ALL_RANKS)) for j in range(i, len(ALL_RANKS))]))

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
            for hand in self.trans_hands:
                #values = [self.value_ranks[c[0]] for c in hand]
                values = [ALL_RANKS.index(c[0]) for c in hand]
                self.hand_values[hand] = (2*max(values) + min(values)
                                        + (PREFLOP_PAIR_VALUE if values[0] == values[1] else 0)
                                        + (PREFLOP_SUITED_VALUE if hand[0][1] == hand[1][1] else 0))
            self.sorted_hands = sorted(self.trans_hands, key=lambda l: self.hand_values[l], reverse=True)
            self.evidence_updated = False
        self.sorted_street = 0

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
            order_ensemble = monte_carlo_perms() # refresh ensemble
            print('[...]')
            for order in order_ensemble[-10:]:
                print(order)

            self.value_ranks = value_ranking(order_ensemble)
            self.trans_hands = translate_hands(self.value_ranks, HANDS)
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
        my_hand_trans = [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in my_hand]
        my_hand_trans_e7 = convert_e7(my_hand_trans)
        trans_board = [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in board]
        board_trans_e7 = convert_e7(trans_board)

        print(my_hand, board)
        print(my_hand_trans, trans_board, '(translated)')

        # raise sizes
        # these are parameters to be tweaked (maybe by stats on opponent)
        if street == 0:
            RAISE_SIZE = 1 # raise (1x)pot
        else:
            RAISE_SIZE = 0.75 # bet 3/4 pot


        if my_pip == 1 and street == 0:
            pot_odds = 0.4
            raise_odds = 0.45
        else:
            pot = 2*my_contribution
            bet = opp_pip - my_pip

            pot_odds = bet / (pot + 2*bet)

            raise_amt = int(RAISE_SIZE * pot) + opp_pip

            raise_odds = raise_amt / (pot + 2*raise_amt)

            print('pot odds = ', pot_odds, 'raise odds = ', raise_odds)

        equity = eval7.py_hand_vs_range_monte_carlo(my_hand_trans_e7, self.random_hand, board_trans_e7, self.N_MCMC)

        if equity > raise_odds and opp_contribution < STARTING_STACK:
            raise_size = min(int(opp_pip + RAISE_SIZE*2*opp_contribution), my_pip + my_stack)
            print('RAISE:', raise_size)
            return RaiseAction(raise_size)
        elif equity > pot_odds and opp_pip > my_pip:
            print('CALL')
            return CallAction()
        elif my_pip == opp_pip:
            print('CHECK')
            return CheckAction()
        else:
            return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
