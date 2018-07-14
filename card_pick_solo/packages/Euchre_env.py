import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

suits_analog = ['h','d','c','s']
values_analog = ['9','t','j','q','k','a']
suits = ['104', '100', '099', '115']
values = ['057', '116', '106', '113', '107', '097']

alts = {'115': '099', '099': '115', '104': '100', '100': '104'}

# tranks = 'akqt9'
tranks = ['097', '107', '113', '116', '057']
# cranks = 'akqjt9'
cranks = ['097', '107', '113', '106', '116', '057']


class Euchre(object):

    def __init__(self):
        self.deck =[]
        self.state = []
        self.trump = []
        self.play_space = 3
        self.round = 0

    @staticmethod
    def winner(cards, trump):
        if '106' + trump in cards:
            return cards.index('106' + trump)
        if '106' + alts[trump] in cards:
            return cards.index('106' + alts[trump])
        for r in tranks:
            if r + trump in cards:
                return cards.index(r + trump)

        lead = cards[0][-3:]
        for r in cranks:
            if r + lead in cards:
                return cards.index(r + lead)

    @staticmethod
    def theoretical_win(board, hand, trump):
       #print("board:", board, "hand:", hand, "trump:", trump)
        hi_trump_b = 0
        hi_trump_h = 0
        hi_card_h = 0
        hi_card_b = 0

        #print("\nchecking for bowers in hand")
        if '106' + trump in hand:
            return True
        if '106' + alts[trump] in hand:
            return True

        #print("\nno bowers checking for high trump")

        for card in reversed(tranks):
            # find the highest trump on the board
            if card + trump in board:
                hi_trump_b = tranks.index(card)
            # find the highest trump in hand
            if card + trump in hand:
                hi_trump_h = tranks.index(card)
        #print("highest trump on board:", tranks[hi_trump_b], "highest trump in hand:", tranks[hi_trump_h])
        if hi_trump_h < hi_trump_b:
            return True

        lead = board[0][-3:]
        #print("\nno trump checking for high card, lead is:", lead)
        for card in reversed(cranks):
            # find the highest non-trump on the board
            if card + lead in board:
                hi_card_b = cranks.index(card)
            # find the highest non-trump in hand
            if card + lead in hand:
                hi_card_h = cranks.index(card)

        #print("highest card on board:", cranks[hi_card_b], "highest card in hand:", cranks[hi_card_h])
        if hi_card_h < hi_card_b:
            return True

        return False

    def step(self, action):
        win = False
        reward = 0
        self.round += 1
        possible = True
        done = False

        # create state divisions
        board = self.state[0:3]
        hand = self.state[3:-1]
        trump = self.state[-1:][0]
        picked_card = hand[action]
        print("PICKED CARD:", picked_card)
        # determine winner
        if picked_card != 0:
            idx = self.winner(board+[picked_card], trump)
            if idx == self.play_space:
                win = True
            else:
                win = False

        # determine reward
        #print(win)
        if picked_card == '000000':
            reward = -2
        else:
            if win:
                reward = 2
            elif not win:
                possible = self.theoretical_win(board, hand, trump)
                #print(possible)
                if possible:
                    reward = -1
                else:
                    reward = 0

        # prep next state package
        next_state = self.deck[0:3]
        self.deck = self.deck[3:]
        hand[action] = '000000'

        next_state = next_state + hand + [trump]
        self.state = next_state

        if self.round >= 5:
            done = True
            self.round = 0

        return next_state, reward, done


    def reset(self):
        self.deck = []
        self.create_deck()
        return self.deal()

    def create_deck(self):
        for s in suits:
            self.trump.append(s)
            for v in values:
                self.deck.append(v + s)
        random.shuffle(self.deck)
        random.shuffle(self.trump)

    def deal(self):
        self.state = self.deck[0:8]
        self.deck = self.deck[8:]
        self.state.append(self.trump[0])
        return self.state
