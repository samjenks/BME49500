import torch
import random
from torch import FloatTensor as tensor

suits_analog = ['h','d','c','s']
values_analog = ['9','t','j','q','k','a']
suits = ['104', '100', '099', '115']
values = ['057', '116', '106', '113', '107', '097']

file = open("possible_states.txt" , 'w')
possibilities = []

for i in range(10000):
    # make deck
    deck = []
    for s in suits:
        for v in values:
            deck.append(v + s)
    random.shuffle(deck)

    # deal
    while len(deck) > 4:
        state = []
        for j in range(8):
            state.append(deck.pop(0)+" ")

        random.shuffle(suits)
        state.append(suits[0])

        if state not in possibilities:
            print(state)
            possibilities.append(state)
            file.writelines(state)
            file.write("\n")




file.close()
