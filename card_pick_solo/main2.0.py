import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
from torch import FloatTensor as tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from card_pick_solo.packages.model import ModelStrat_simple as model

net = model()
optimizer = optim.RMSprop(net.parameters())
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
suits_analog = ['h','d','c','s']
values_analog = ['9','t','j','q','k','a']
suits = ['104', '100', '099', '115']
values = ['057', '116', '106', '113', '107', '097']

alts = {'115': '099', '099': '115', '104': '100', '100': '104'}

# tranks = 'akqt9'
tranks = ['097', '107', '113', '116', '057']
# cranks = 'akqjt9'
cranks = ['097', '107', '113', '106', '116', '057']


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(10000)


def winner(cards, trump):
    cards = cards.tolist()

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


def play_card_func(outside_state):


    temp = np.array(outside_state, dtype='str')
    hand = temp[...,-6:-1]

    inside_state = np.array(outside_state).astype(np.float)
    input = Variable(tensor(inside_state))
    output = net.forward(input)

    val, idx = torch.max(output, 1)
    #print(output)

    idx = idx.data.view(BATCH_SIZE, 1)
    #print(idx)

    card = []
    for j in range(len(hand)):
        card.append(hand.item((j,idx[j][0])))
    card = np.array(card, dtype='str')

    #print(hand[0], hand.shape)
    #print(card)

    return temp, card

    """state = torch.cat([tensor([float(i) for i in outside_state]).view(1,4),
                           tensor([float(i) for i in self.hand]).view(1,5), tensor([float(trump)]).view(1,1)], 1)
        self.var_strat.data = state
        card_to_play, self.hidden = self.strat.forward(self.var_strat, self.hidden)
        val, idx = torch.max(card_to_play.view(1,5), 1)
        idx = idx.data.view(1,1)[0][0]
        outgoing_card = self.hand[idx]
        self.hand[idx] = 0
        return outgoing_card"""

    #'097100' '106115' '057115' '116099' '116104'

def convert_dec(str):
    val = float(str[0])
    suit = int(str[1])
    print(val, suit)

def reward_mech(state, action):
    #print(state, state.shape,"\n\n", action, action.shape)
    board = state[...,0:3]
    rewards = []
    #print(len(state[0])-1)
    i = 0
    for board_state in board:
        win = winner(np.append(board_state, action[i]), state[i][len(state[i])-1])
        i +=1
        if win == 3:
            rewards.append(1)
        else:
            rewards.append(-1)

    return np.array(rewards)


def expected(hand, board):
    lead = board[...,0:1]




def optimize_model(state, action, rewards):
    """if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))


    #state_batch = batch.state
    #action_batch = batch.action
    #reward_batch = batch.reward

    print(reward_batch)
    """
    board = state[...,0:3]
    hand = state[...,-6:-1]

    expected_rewards = Variable(torch.ones((BATCH_SIZE,1)), requires_grad=True)
    loss = F.smooth_l1_loss(Variable(rewards, requires_grad=True), expected_rewards)
    optimizer.zero_grad()
    loss.backward()
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



if __name__ == '__main__':

    states = []
    lines = [line.rstrip('\n') for line in open('packages/possible_states.txt')]
    for line in lines:
        states.append(line.split(" "))



    num_episodes = int(len(states) / BATCH_SIZE)
    for i_episode in range(num_episodes):
        # Initialize the environment and state

        random.shuffle(states)
        batch = []
        for x in range(BATCH_SIZE):
            batch.append(states[x])


        # Select and perform an action
        state, action = play_card_func(batch)

        # determine reward
        reward = reward_mech(state, action)

        reward = tensor(np.array([reward])).view(BATCH_SIZE, 1)



        # Store the transition in memory
        memory.push(state, action, reward)


        # Perform one step of the optimization (on the target network)
        optimize_model(state, action, reward)

