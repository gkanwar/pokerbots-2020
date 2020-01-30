import eval7

data = open('preflop_odds.txt','r').readlines()

hand_dict = {}
for i in data[1:]:
	line = i.strip().split(',')
	hand = line[0]
	odds = [float(j) for j in line[1:]]
	hand_dict[hand] = odds
#print(hand_dict)

hand_fractions = [float(i) for i in data[0].strip().split(',')]

def get_fraction_index(fraction):
	for i in range(len(hand_fractions)):
		if int(hand_fractions[i]) / 100.0 > fraction:
			return i
	return i

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
	return eq

if __name__ == '__main__':

	hand = [eval7.Card('3c'),eval7.Card('2d')]
	print(get_equity(hand))

	hand = [eval7.Card('3c'),eval7.Card('2c')]
	print(get_equity(hand,4))

	hand = [eval7.Card('2c'),eval7.Card('2d')]
	print(get_equity(hand,5))
