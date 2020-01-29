import copy
import numpy as np
import sys
import time
import traceback as tb

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
            j1,j2 = order[i], order[i+1]
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
            if j1 < j2 and np.random.random() < bwd_prob:
                order[i], order[i+1] = order[i+1], order[i]
            elif j2 < j1 and np.random.random() < fwd_prob:
                order[i], order[i+1] = order[i+1], order[i]
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
print('[...]')
for order in order_ensemble[-10:]:
    print(order)
