import eval7
import numpy as np


######################################################
#load in the preflop equity data
data = open('preflop_odds.txt','r').readlines()

hand_dict = {}
for i in data[1:]:
	line = i.strip().split(',')
	hand = line[0]
	odds = [float(j) for j in line[1:]]
	hand_dict[hand] = odds

hand_fractions = [float(i) for i in data[0].strip().split(',')]
######################################################

######################################################
#hacky code to build ranges based on hand odds
#odds from https://www.cardschat.com/poker-starting-hands.php
data = open('hand_odds_table.txt','r').readlines()

cards = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']

hand_odds = np.zeros((13,13))
hand_odds_order = []
for i in range(13):
	for j in range(13):
		line = data[15*i + j + 1]
		value = line.strip()[-8:-6]
		hand_odds[i][j] = float(value)
		if i == j:
			hand_odds_order.append([float(value),cards[i] + cards[j],6])
		elif i < j:
			hand_odds_order.append([float(value),cards[i] + cards[j] + 's',4])
		else:
			hand_odds_order.append([float(value),cards[j] + cards[i] + 'o',12])

hand_odds_order.sort()
hand_odds_order = hand_odds_order[::-1]


#vranges = [eval7.HandRange(''.join([hand_odds_order[z][1] + ',' for z in range(i)])[:-1]) for i in range(1,len(hand_odds_order)+1)]
vrange_strings = [','.join([hand_odds_order[z][1] for z in range(i)]) for i in range(1,len(hand_odds_order)+1)]

######################################################



def get_fraction_index(fraction):
	for i in range(len(hand_fractions)):
		if int(hand_fractions[i]) / 100.0 > fraction:
			return i
	return i

vranges = [-1] * len(vrange_strings)
#returns eval7 HandRange object
def get_v_range(fraction):
	ind = get_fraction_index(fraction)

	#check if we haven't make the vrange obect yet
	if vranges[ind] == -1:
		vranges[ind] = eval7.HandRange(vrange_strings[ind])

	return vranges[ind]

def get_preflop_equity(hand,fraction):
	c1 = str(hand[0])
	c2 = str(hand[1])

	ind = get_fraction_index(fraction)
	#print(ind)
	#check if same card
	if c1[0] == c2[0]:
		dict_hand = c1[0] + c2[0]
		eq = hand_dict[dict_hand][ind]

	#check if suited
	elif c1[1] == c2[1]:
		try:
			dict_hand = c1[0] + c2[0] + 's'
			eq = hand_dict[dict_hand][ind]
		except:
			dict_hand = c2[0] + c1[0] + 's'
			eq = hand_dict[dict_hand][ind]

	#offsuit
	else:
		try:
			dict_hand = c1[0] + c2[0] + 'o'
			eq = hand_dict[dict_hand][ind]
		except:
			dict_hand = c2[0] + c1[0] + 'o'
			eq = hand_dict[dict_hand][ind]
	return eq / 100

if __name__ == '__main__':

	hand = [eval7.Card('3c'),eval7.Card('2d')]
	print(get_equity(hand))

	hand = [eval7.Card('3c'),eval7.Card('2c')]
	print(get_equity(hand,4))

	hand = [eval7.Card('2c'),eval7.Card('2d')]
	print(get_equity(hand,5))
