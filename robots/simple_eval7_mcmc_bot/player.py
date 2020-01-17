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

from perm import *

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
        pass

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
        pass

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
        #my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        #opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        #my_stack = round_state.stacks[active]  # the number of chips you have remaining
        #opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        #continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        #my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        #opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        #if RaiseAction in legal_actions:
        #    min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
        #    min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
        #    max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

        def check_fold():
            if CheckAction in legal_actions: return CheckAction()
            return FoldAction()

        # Simple hardcoded / eval7 strategy.
        street = round_state.street
        my_hand = round_state.hands[active]
        board = round_state.deck[:street]
        
        # Pre-flop, raise pairs and suited for example
        PREFLOP_PAIR_SUITED_VALUE = 1.0
        if street == 0:
            if my_hand[0][0] == my_hand[1][0] or my_hand[0][1] == my_hand[1][1]:
                mean_hand_val = PREFLOP_PAIR_SUITED_VALUE
            else:
                mean_hand_val = 0.0
        # Post-flop, need to do some sims
        else:
            hand_ensemble = []
            board_ensemble = []
            for order in order_ensemble:
                hand_ensemble.append([order[ALL_RANKS.index(card[0])] + card[1] for card in my_hand])
                board_ensemble.append([order[ALL_RANKS.index(card[0])] + card[1] for card in board])
            # simply eval7 each translation
            TRIPS_UP_VALUE = 1.0
            PAIR_UP_VALUE = 0.75
            hand_vals = []
            for hand, board in zip(hand_ensemble, board_ensemble):
                code = eval7.evaluate([eval7.Card(s) for s in hand + board])
                handtype = code >> 24
                if handtype >= 3: # we have something trips or higher
                    hand_vals.append(TRIPS_UP_VALUE)
                elif handtype >= 1: # we have something pair or higher
                    hand_vals.append(PAIR_UP_VALUE)
                else:
                    hand_vals.append(0.0)
            mean_hand_val = np.mean(hand_vals)
        
        # bet our value (this is basically bogus)
        # value * (2*x + pot) = x
        # (1 - 2*value) x = value*pot
        # x = pot * value / (1-2*value).
        # Note: for value > 0.5, our bets go through infinty and negative. Instead,
        # run this all through a tanh function for stability (more bogus).
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        pot = 2*STARTING_STACK - my_stack - opp_stack
        bet = int(pot * mean_hand_val / (1-np.tanh(mean_hand_val)))
        min_raise, max_raise = round_state.raise_bounds()
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        call_cost = min_raise - my_pip
        if bet < min_raise:
            return check_fold()
        elif max_raise == 0:
            soft_assert(CheckAction in legal_actions)
            return CheckAction()
        elif RaiseAction in legal_actions:
            return RaiseAction(min(max_raise, bet + my_pip))
        else:
            soft_assert(CallAction in legal_actions)
            return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
