import eval7
import matplotlib.pyplot as plt
import numpy as np

N = 1000
VALUES = [str(i) for i in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']

# build all possible hands
# pairs:
hands = [[s+s, 6, 0] for s in VALUES]
# non-pairs:
hands += [[VALUES[i]+VALUES[j]+s, 12 if s=='o' else 4, 0] for i in range(len(VALUES)) for j in range(i+1, len(VALUES)) for s in ['o','s']]

total_hands = sum(x[1] for x in hands)
random_hand = eval7.HandRange(','.join(x[0] for x in hands))

print('Calculating hand equities...')
for i, hand in enumerate(hands):
    equities = eval7.py_all_hands_vs_range(eval7.HandRange(hand[0]), random_hand, [], N)
    hand[2] = np.mean([equities[v] for v in equities])

hands = sorted(hands, key=lambda x: x[2], reverse=True)

f = np.zeros(len(hands))
equities = np.zeros(len(hands))

print('Calculating range equities...')
cur_fraction = 0
for i, hand in enumerate(hands):
    cur_fraction += hand[1]/total_hands
    f[i] = cur_fraction
    equities[i] = np.average([x[2] for x in hands[:i+1]], weights=[x[1] for x in hands[:i+1]])

stack = 200
sb = 1
bb = 2
bb_ev = (f*( # fraction we call the shove
            equities*stack # our share of the pot if we call and win
            +
            (1-equities)*(-stack) # losses if we call and lose
           )
         +
         (1-f)*(-bb) # fraction we fold and relinquish bb
        )
sb_raise = 0.85
sb_raise_size = 6
sb_ev = (
         sb_raise*( # fraction we raise our sb
             f/sb_raise*( # fraction we call (dividing by sb_raise so that we plot this against % of total range (not % of raising range
                 equities*stack # call/win
                 +
                 (1-equities)*(-stack) # call/lose
             )
             +
             (1-f/sb_raise)*(-sb_raise_size) # fraction we raise/fold
         )
         +
         (1-sb_raise)*(-sb) # fraction we fold and lose our sb
        )

plt.figure(figsize=(11,7))
plt.plot(f, equities, '.-')
plt.grid(which='both')
plt.xlabel('Fraction of hands')
plt.ylabel('Equity against random hand')
plt.gca().autoscale(axis='x', tight=True)
plt.tight_layout()

plt.figure(figsize=(11,7))
plt.plot(f, bb_ev, '.-', label='BB EV')
i_sb_end = np.argmin(np.abs(f-sb_raise))
plt.plot(f[:i_sb_end], sb_ev[:i_sb_end], '.-', label='SB EV')
plt.grid(which='both')
plt.xlabel('Fraction of hands')
plt.ylabel('EV [$]')
plt.legend(loc='best')
plt.gca().autoscale(axis='x', tight=True)
plt.show()
