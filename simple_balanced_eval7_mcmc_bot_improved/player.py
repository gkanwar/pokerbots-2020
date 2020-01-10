'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import copy
import eval7
import numpy as np
import time
import traceback as tb
import sys
from math import ceil

DEBUG_MODE = False
ALL_RANKS = list(map(str, range(2, 9+1))) + ['T', 'J', 'Q', 'K', 'A']
GEOM_P = 0.25

# only hard fail in debug mode
def soft_assert(q):
    if DEBUG_MODE:
        assert q
    elif not q:
        print('WARNING: soft_assert failed!')
        tb.print_stack(file=sys.stdout)

def get_best_hand_type(hand, board):
    suits = []
    numbers = []
    for c in hand:
        suits.append(c[1])
        numbers.append(c[0])
    for c in board:
        suits.append(c[1])
        numbers.append(c[0])
    # four-of-a-kind?
    num_vals, num_counts = np.unique(numbers, return_counts=True)
    if np.any(num_counts == 4):
        return 'four_kind', num_vals[num_counts==4][0]
    # full-house?
    if sorted(num_counts)[-1] >= 3 and sorted(num_counts)[-2] >= 2:
        return 'full_house', None # TODO: give rank info
    # flush?
    suit_vals, suit_counts = np.unique(suits, return_counts=True)
    if np.any(suit_counts >= 5):
        return 'flush', suit_vals[suit_counts>=5][0]
    # set?
    if np.any(num_counts == 3):
        # TODO: what if multiple sets available?
        soft_assert(np.sum(num_counts == 3) == 1)
        return 'set', num_vals[num_counts==3][0]
    # pair?
    if np.any(num_counts == 2):
        if np.sum(num_counts == 2) == 1:
            return 'pair', num_vals[num_counts==2][0]
        else:
            return 'two_pair', num_vals[num_counts==2]
    soft_assert(np.all(num_counts == 1))
    return 'high_or_unknown', None

def test_get_best_hand_type():
    print('[RUNNING test_get_best_hand_type]')
    test = get_best_hand_type(['4c', '8h'], ['Td', '9h', '7s', '6d', '8d'])
    print(test)
    assert test == ('pair', '8')
    test = get_best_hand_type(['4c', '8h'], ['Td', '9h', '7s', '4d', '8d'])
    print(test)
    assert test[0] == 'two_pair'
    assert set(test[1]) == set(['4', '8'])
    test = get_best_hand_type(['4c', '8c'], ['Tc', '9c', '7c', '6d', '5d'])
    print(test)
    assert test == ('flush', 'c')
    test = get_best_hand_type(['4c', '8c'], ['Tc', '9c', '7c', '6c', '5d'])
    print(test)
    assert test == ('flush', 'c')
    test = get_best_hand_type(['4c', '8h'], ['Td', '9h', '8s', '6d', '8d'])
    print(test)
    assert test == ('set', '8')
    test = get_best_hand_type(['4c', '8h'], ['Td', '4h', '8s', '6d', '8d'])
    print(test)
    assert test == ('full_house', None) # FORNOW
    test = get_best_hand_type(['8c', '8h'], ['Td', '9h', '8s', '6d', '8d'])
    print(test)
    assert test == ('four_kind', '8')
    print('[PASSED test_get_best_hand_type]')
# if __name__ == '__main__': test_get_best_hand_type()


# Evidence object for perm^{-1}(a) > perm^{-1}(b).
class Evidence(object):
    def __init__(self, a, b, hand0, hand1, board, winner):
        self.a = a
        self.b = b
        self.hand0 = hand0
        self.hand1 = hand1
        self.board = board
        self.winner = winner
    def __str__(self):
        return 'Evidence({} > {})'.format(self.a, self.b)
    __repr__ = __str__
# Build evidence (or None) on permutation based on possible showdown.
def get_perm_evidence(terminal_state, active):
    start = time.time()
    last_state = terminal_state.previous_state
    if len(last_state.hands[1-active]) == 0:
        print('No show, no info.')
        return None
    hand0, hand1 = last_state.hands
    board = last_state.deck
    assert len(board) == 5 # reached showdown
    if terminal_state.deltas[0] > 0:
        winner = 0
    elif terminal_state.deltas[1] > 0:
        winner = 1
    else:
        print('Chop, no info.')
        return None
    print('Getting showdown evidence:')
    print('hand0\thand1\tboard')
    print('{}\t{}\t{}'.format(' '.join(hand0), ' '.join(hand1), ' '.join(board)))
    print('winner = {}'.format(winner))
    # we only have info if the numerical values affect the winner
    # it's very rare for a sneaky straight to appear and mess up our info, but
    # we should handle possible conflicting evidence down the line.
    type0, val0 = get_best_hand_type(hand0, board)
    type1, val1 = get_best_hand_type(hand1, board)
    win_val = [val0, val1][winner]
    lose_val = [val0, val1][1-winner]
    if type0 == type1:
        if (type0 == 'pair' or type0 == 'set') and win_val != lose_val:
            evi = Evidence(win_val, lose_val, hand0, hand1, board, winner)
            print('Got evidence from showdown: {}', evi)
            return evi
        # TODO: other types of evidence?
    print('No evidence from this showdown.')
    return None

# partial order graph given evidences
class PartialOrderGraph:
    def __init__(self):
        self.edges = {}
        self.nodes = ALL_RANKS[:]
        assert len(self.nodes) == 13
        self.children = {k: [] for k in self.nodes}
    def remove_edge(self, edge):
        del self.edges[edge]
        self.children[edge[0]].remove(edge[1])
    def make_edge(self, edge, *, weight):
        self.edges[edge] = weight
        self.children[edge[0]].append(edge[1])
    def remove_leaf(self, node):
        assert len(self.children[node]) == 0
        del self.children[node]
        self.nodes.remove(node)
        keys = list(self.edges.keys())
        for k in keys:
            if node in k:
                self.remove_edge(k)
    def add_evidence(self, evidence):
        fwd_key = (evidence.a, evidence.b)
        bwd_key = (evidence.b, evidence.a)
        if fwd_key in self.edges:
            self.edges[fwd_key] += 1
        elif bwd_key in self.edges:
            val = self.edges[bwd_key] - 1
            if val == 0:
                self.remove_edge(bwd_key)
            elif val < 0:
                self.remove_edge(bwd_key)
                self.make_edge(fwd_key, weight=-val)
            else:
                self.edges[bwd_key] = val
        else:
            self.make_edge(fwd_key, weight=1)
    def __str__(self):
        ss = ['digraph partial_order_graph {']
        for a,b in self.edges:
            ss.append(f'\t{a} -> {b} [label={self.edges[(a,b)]}];')
        for n in self.nodes:
            ss.append(f'\t{n};')
        ss.append('}')
        return '\n'.join(ss)

partial_order = PartialOrderGraph()

def detect_cyclic_edge(graph):
    nodes, edges = graph.nodes, graph.edges
    seen = set()
    for n in nodes:
        if n in seen: continue
        queue = [(n, [n])]
        while len(queue) > 0:
            n, path = queue.pop()
            seen.add(n)
            for n2 in nodes:
                key = (n,n2)
                if key not in edges: continue
                if n2 in path: return key # cycle!
                if n2 in seen: continue
                queue.append((n2, path+[n2]))
    return None

def get_leaves(g):
    leaves = []
    for n in g.nodes:
        if len(g.children[n]) == 0:
            leaves.append(n)
    return leaves

# Draw samples from (approx) posterior distributions of perms given by partial
# order graph that we have so far. There will likely be mistakes due to missed
# straights. For now, we break cycles by randomly deleting edges (inversely
# weighted by evidence count), and prior to that we randomly flip edges again
# inversely weighted by evidence count. This introduces noise, but prevents
# confidently betting into a mistake.
def mcmc_update(order, g):
    # assumes g is a DAG, and order already satisfies it.
    # Gibbs resample nearest neighbor order, if valid to reorder
    done = False
    while not done:
        seeds = np.random.randint(len(g.nodes)-1, size=8)
        for i in seeds: # we are proposing to swap VALUE i and i+1
            j1,j2 = order.index(i), order.index(i+1)
            fwd_key = (g.nodes[j1], g.nodes[j2])
            bwd_key = (g.nodes[j2], g.nodes[j1])
            # check reorder validity
            if fwd_key in g.edges or bwd_key in g.edges: continue
            # get relative probs of each order (i,i+1) or (i+1,i):
            # The only difference in probability is when the first one of the
            # pair is assigned. No other orderings change, but we either choose
            # immediately or skip with probability (1-GEOM_P), then choose.
            fwd_prob = 1
            bwd_prob = 1-GEOM_P
            if np.random.random() < bwd_prob / (fwd_prob + bwd_prob):
                order[j1], order[j2] = order[j2], order[j1]
            done = True
            break
            
def monte_carlo_perms(*, n=100, n_skip=100, n_therm=1000):
    # Step 1: manipulate graph into DAG
    decycle_g = copy.deepcopy(partial_order)
    while True:
        e = detect_cyclic_edge(decycle_g)
        if e is None: break
        decycle_g.remove_edge(e) # FORNOW: just kill the edge
    print('De-cycled graph:')
    print(decycle_g)
    # Step 2: sample according to a DAG, by iteratively sampling the available
    # leaves with the correct prior probabilities.
    ensemble = []
    # make init order satisfying constraints
    order = []
    g = copy.deepcopy(decycle_g)
    while True:
        leaves = get_leaves(g)
        order.extend(leaves)
        for l in leaves: g.remove_leaf(l)
        if len(g.nodes) == 0: break
    print(f'face value order: {order}')
    order = list(map(ALL_RANKS.index, order))
    order = list(map(order.index, range(len(ALL_RANKS)))) # reverse the mapping
    print(f'initial order = {order}')
    for i in range(-n_therm, n_skip*n):
        mcmc_update(order, decycle_g)
        # Need to use MCMC, since direct sampling seems intractable.
        if i >= 0 and (i+1)%n_skip == 0:
            ensemble.append([ALL_RANKS[i] for i in order])
        # NOTE: this was not the right interpretation of the prior sampling procedure!
        # remaining = copy.deepcopy(partial_order.nodes)
        # g = copy.deepcopy(partial_order)
        # order = []
        # while len(remaining) > 0:
        #     leaves = get_leaves(g)
        #     inds = list(map(remaining.index, leaves))
        #     # resummed probabilities, including infinite wraps
        #     # p_i = geom_p^{i+1} sum_{j=0}^{\infty} geom_p^{# remaining * j}
        #     #     = geom_p^{i+1} / (1 - geom_p^{# remaining})
        #     n_rem = len(remaining)
        #     ps = np.array([geom_p ** (i+1) / (1 - geom_p**n_rem) for i in inds])
        #     ps /= np.sum(ps) # normalize
        #     order.append(np.random.choice(leaves, p=ps))
        #     remaining.remove(order[-1])
        #     g.remove_leaf(order[-1])
        # ensemble.append(order)
    return ensemble

def value_ranking(ensemble):
    ranks = dict((c,0) for c in ALL_RANKS)
    for order in order_ensemble:
        for i, o in enumerate(order):
            ranks[o] += i
    return dict((c,k) for k,c in enumerate(sorted(ranks.keys(), key=lambda x: ranks[x])))

## NOTE: This global ensemble will be refreshed as new evidence comes in. We can
## use ensemble statistics to choose actions fairly confidently.
order_ensemble = monte_carlo_perms()

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
        self.hand_values = dict((hand, 0) for hand in HANDS)
        self.sorted_hands = []
        self.value_ranks = value_ranking(order_ensemble)
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
            for hand in HANDS:
                values = [self.value_ranks[c[0]] for c in hand]
                self.hand_values[hand] = (2*max(values) + min(values)
                                        + (PREFLOP_PAIR_VALUE if values[0] == values[1] else 0)
                                        + (PREFLOP_SUITED_VALUE if hand[0][1] == hand[1][1] else 0))
            self.sorted_hands = sorted(HANDS, key=lambda l: self.hand_values[l], reverse=True)
            self.evidence_updated = False
        self.sorted_street = 0
        self.range = self.sorted_hands

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

        # re-sort range based on board texture
        if street > 0 and street != self.sorted_street:
            hand_values = {}

            trans_board = [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in board]
            new_range = []
            for hand in self.range:
                if not hand[0] in board and not hand[1] in board:
                    trans_hand = [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in hand]
                    value = eval7.evaluate([eval7.Card(s) for s in trans_hand + trans_board])

                    # don't value straights at all
                    if (value >> 24) == 4:
                        # remove single cards so we only have pair/2p/trips ranking left
                        counts = np.unique([x[0] for x in trans_hand + trans_board], return_counts=True)
                        duplicates = [counts[0][i] for i in range(len(counts[0])) if counts[1][i] > 1]

                        # new value based on just duplicate cards
                        value = eval7.evaluate([eval7.Card(s) for s in trans_hand + trans_board if s[0] in duplicates])

                    hand_values[hand] = value

                    new_range += [hand]

            self.range = sorted(new_range, key=lambda l: hand_values[l], reverse=True)
            self.sorted_street = street

            print('====================')
            print('Ranks:', self.value_ranks)
            print(my_hand, board)
            print([ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in my_hand], [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in board], '(translated)')
            # print('Top 20 hands in range:', [(h, hand_values[h]) for h in self.range[:20]])
            # print('Range: ', self.range)

        # rank current hand in our range
        N = len(self.range)
        hand_position = self.range.index(tuple(my_hand)) / N

        print('Hand %:', hand_position, my_hand, [ALL_RANKS[self.value_ranks[c[0]]]+c[1] for c in my_hand], self.value_ranks)
        print(raise_range, '(raise)', mdf, '(defend)', bluff_range, '(bluff)')

        # determine whether hand is a raise/call/bluff-raise/check/fold

        # currently commented out range updates; to be "unexploitable" our ranges should be
        # restricted at each decision (otherwise can be overbluffed if we show weakness), but
        # that results in us overvaluing weak hands.
        # e.g. if we check flop, we eliminate all ~pair+ holdings
        #      then on turn if we bet the top 33% of our range, we'll have to start value-betting high card hands
        # to do it properly we need some "slowplay" range to protect our delayed value hands
        if (hand_position < raise_range or mdf < hand_position < bluff_range) and opp_contribution < STARTING_STACK:
            raise_size = min(int(opp_pip + RAISE_SIZE*2*opp_contribution), my_pip + my_stack)

            if street == 0:
                self.range = self.range[:ceil(N*raise_range)] + self.range[int(N*mdf):int(ceil(N*bluff_range))]

            print('RAISE:', raise_size)
            return RaiseAction(raise_size)
        elif hand_position < mdf and opp_pip > my_pip:
            if street == 0:
                self.range = (self.range[:int(N*raise_range)] if opp_pip == my_pip+my_stack else []) + self.range[int(N*raise_range):int(ceil(N*mdf))]

            print('CALL')
            return CallAction()
        elif my_pip == opp_pip:
            if street == 0:
                self.range = self.range[int(N*raise_range):int(ceil(N*mdf))] + self.range[int(N*bluff_range):]

            print('CHECK')
            return CheckAction()
        else:
            return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
