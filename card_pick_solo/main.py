#!/usr/local/bin/python3.6

import random
import torch
import numpy as np
from torch import FloatTensor as tensor
import torch.optim as optim

from card_pick_solo.packages.model import ModelStrat_simple as model
from torch.autograd import Variable


suits_analog = ['h','d','c','s']
values_analog = ['9','t','j','q','k','a']
suits = ['104', '100', '099', '115']
values = ['057', '116', '106', '113', '107', '097']

alts = {'115': '099', '099': '115', '104': '100', '100': '104'}

# tranks = 'akqt9'
tranks = ['097', '107', '113', '116', '057']
# cranks = 'akqjt9'
cranks = ['097', '107', '113', '106', '116', '057']

net = model()
optimizer = optim.Adam(net.parameters(), lr=1e-2)


def winner(cards, trump):
    if '106' + trump in cards:
        return cards.index('106' + trump)
    if '106' + alts[trump] in cards:
        return cards.index('106' + alts[trump])
    for r in tranks:
        if r + trump in cards:
            return cards.index(r + trump)

    for r in cranks:
        if r + cards[0][:-3] in cards:
            return cards.index(r + cards[0][:-3])
    #return random.randint(0 ,3)


def play_card_func(outside_state, trump):
    tricks = 0

    board = [outside_state[0], outside_state[1], outside_state[2]]
    hand = outside_state[-6:-1]
    #print(board, hand, trump)

    outside_state = np.array(outside_state).astype(np.float)
    input = Variable(tensor(outside_state))
    output = net.forward(input)
    val, idx = torch.max(output, 0)

    card = hand[idx.data.view(1, 1)[0][0]]
    board.append(card)
    win = winner(board, trump)
    if win == 3:
        tricks = 1
    else:
        tricks = -1

    return tricks, output

    """state = torch.cat([tensor([float(i) for i in outside_state]).view(1,4),
                           tensor([float(i) for i in self.hand]).view(1,5), tensor([float(trump)]).view(1,1)], 1)
        self.var_strat.data = state
        card_to_play, self.hidden = self.strat.forward(self.var_strat, self.hidden)
        val, idx = torch.max(card_to_play.view(1,5), 1)
        idx = idx.data.view(1,1)[0][0]
        outgoing_card = self.hand[idx]
        self.hand[idx] = 0
        return outgoing_card"""


def convert_dec(str):
    val = float(str[0])
    suit = int(str[1])
    print(val, suit)


def reward(rewards, output):
    optimizer.zero_grad()
    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    loss = tensor([rewards.mean()])
    output = Variable(loss, requires_grad=True)
    output.backward()
    optimizer.step()


def percentage(rewards):
    total = len(rewards)
    wins = sum(i > 0 for i in rewards)
    print(wins/total)


if __name__ == '__main__':

    # pre-processing
    states = []
    lines = [line.rstrip('\n') for line in open('packages/possible_states.txt')]
    for line in lines:
        states.append(line.split(" "))

    #print(states[0][0:-1], states[0][-1:])

    for i in range(2000):
        random.shuffle(states)

        BATCH_SIZE = 64

        batch = []
        # play card
        for x in range(BATCH_SIZE):
            batch.append(states[x])

        results = []
        for state in batch:
            tricks, output = play_card_func(state, state[-1:][0])
            results.append(tricks)

        # reward
        #print(results)
        reward(tensor(results), output)
        percentage(results)




