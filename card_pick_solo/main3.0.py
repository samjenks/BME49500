import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
from torch import FloatTensor as tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from card_pick_solo.packages.model import ModelStrat_simple as model
from card_pick_solo.packages.Euchre_env import Euchre


def encode(x,l):
    return tensor(np.array(x).astype(np.float))


def OH(x, l):
    suits = {'104':28, '100':29, '099':30, '115':31}
    zero_count = 0
    cards = {'057104':0, '116104':1, '106104':2,'113104':3,'107104':4, '097104':5,
             '057100': 6, '116100': 7, '106100': 8, '113100': 9, '107100': 10, '097100': 11,
             '057099': 12, '116099': 13, '106099': 14, '113099': 15, '107099': 16, '097099': 17,
             '057115': 18, '116115': 19, '106115': 20, '113115': 21, '107115': 22, '097115': 23,
             '000000': 24}
    #print("oh, x:",x)
    idxs = []
    for i in range(0,8):
        if x[i] != '000000':
            idxs.append(cards[x[i]])
        else:
            idxs.append(cards[x[i]]+zero_count)
            zero_count+=1
    idxs.append(suits[x[8]])
    y = torch.LongTensor(idxs).view(1,9)

    #print("OH y:", y)
    one_hot = torch.FloatTensor(1,l)
    return one_hot.zero_().scatter_(1,y,1)


# env = gym.make('FrozenLakeNotSlippery-v0')
env = Euchre()

# Chance of random action
e = 0.1
learning_rate = 0.005
# Discount Rate
gamma = 0.90
# Training Episodes
episodes = 10000
# Max Steps per episode
steps = 20

# Initialize history memory
step_list = []
reward_list = []
loss_list = []
e_list = []
win_list = []

state_space = 24 + 4 + 4 # number of cards + number of possible zeroes + number of suits for trump
action_space = 5


model = model(state_space, action_space)
optimizer = optim.RMSprop(model.parameters(), lr=.005)
zero_plays = 0
other_plays = 0


for i in trange(episodes):
    state = env.reset()
    done = False
    l = 0
    print("\nEpisode:",i)
    while not done:
        state = Variable(OH(state, state_space))
        Q = model(state)
        _, action = torch.max(Q, 1)
        action = action.data[0]
        #print("action:", action)
        new_state, reward, done = env.step(action)
        Q1 = model(Variable(OH(new_state, state_space)))

        if reward == -2:
            zero_plays += 1
        else:
            other_plays += 1

        #print("\nq1:",Q1)
        maxQ1, _ = torch.max(Q1.data, 1)
        maxQ1 = torch.FloatTensor(maxQ1)
        #print("maxq1:", maxQ1)
        targetQ = Variable(Q.data, requires_grad=False).view((1,5))
        targetQ[0, action] = reward + torch.mul(maxQ1, gamma)

        output = model(state)
        train_loss = F.smooth_l1_loss(output, targetQ)
        l += train_loss.data[0]

        model.zero_grad()
        train_loss.backward(retain_graph=True)
        optimizer.step()
        print("curr_state:",state.data.tolist() ,"\nnew_state:",new_state,"\naction:",action,"reward:",reward,
              "loss", train_loss.data[0],"\n")
        state = new_state

print("\n\nzero:", zero_plays, "other:", other_plays)
