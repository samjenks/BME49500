#!/usr/local/bin/python3.6

import random
import torch
from decimal import Decimal
import torch.optim as optim

from Bot.packages.model import ModelCall as mc
from Bot.packages.model import ModelDiscard as dis
from Bot.packages.model import ModelStrat as gameplan
from torch import FloatTensor as tensor
from torch.autograd import Variable

from Bot.packages.model import ModelOrder as mo

suits_analog = ['h','d','c','s']
values_analog = ['9','t','j','q','k','a']
suits = ['104', '100', '099', '115']
values = ['057', '116', '106', '113', '107', '097']


# TODO add not following lead punishment

def random_partners(batches, mode):
    if mode == 'test':
        random.shuffle(batches)
        for player in range(2):
            batches[player].partner = batches[player + 2]
            batches[player + 2].partner = batches[player]

    if mode == 'train':
        for players in batches:
            random.shuffle(players)
            for player in range(2):
                players[player].partner = players[player+2]
                players[player+2].partner = players[player]


def convert_card_2_char(card):
    #print card
    suit = card[-3:]
    val = card[0:3]
    #print suit, val
    return chr(Decimal(val))+chr(Decimal(suit))


def sequence(start, players):
    out = []
    for i in range(4):
        out.append(players[(start+i)%4])
    return out


def first_index_of_suit(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1


def play_hand_train(players, dealers, batch_size):
    d = Deck(players, 'train', batch_size)
    d.make_deck()
    round_seq = sequence(dealers + 1, players)


def play_hand_test(dealer, players):

    d = Deck(players, 'test', 1)
    d.make_deck()
    print (dealer)
    round_seq = sequence(dealer + 1, players)

    deck = d.deal()
    trump = ''
    previous = []
    card = deck.pop()
    print("{0} is the top card".format(convert_card_2_char(card)))
    namer = -1

    # ordering stage
    for i in range(len(round_seq)):
        response = round_seq[i].order(card)
        if response == 1 and trump == '':
            trump = card[-3:]
            namer = round_seq[i]
            previous.append('o')
            discarded = round_seq[i].discard(card, trump)
            print("{0} orders {1} to be trump, and is picking up {2} and discarding {3}"
                  .format(str(round_seq[i].name), chr(Decimal(trump)), convert_card_2_char(card), convert_card_2_char(str(discarded))))
            break

        elif response == 2 and trump == '':
            trump = card[-3:]
            namer = round_seq[i]
            previous.append('o')
            discarded = round_seq[i].discard(card, trump)
            print("{0} orders {1} to be trump, and is picking up {2} and discarding {3} and is going alone"
                  .format(str(round_seq[i].name), chr(Decimal(trump)), convert_card_2_char(card), convert_card_2_char(str(discarded))))
            round_seq[i].partner.playing = False
            break

        else:
            print("{0} has decided to pass".format(str(round_seq[i].name)))
            previous.append('p')

    # calling stage
    if trump == '':
        for i in range(len(round_seq)):
            response = round_seq[i].call()
            # ['104', '100', '099', '115']
            if response == '104' and trump == '':
                trump = '104'
                namer = round_seq[i]
                previous.append('o')
                print("{0} orders {1} to be trump".format(str(round_seq[i].name), chr(Decimal(trump))))
                break

            elif response == '100' and trump == '':
                trump = '1040'
                namer = round_seq[i]
                previous.append('o')
                print("{0} orders {1} to be trump".format(str(round_seq[i].name), chr(Decimal(trump))))
                break

            elif response == '099' and trump == '':
                trump = '099'
                namer = round_seq[i]
                previous.append('o')
                print("{0} orders {1} to be trump".format(str(round_seq[i].name), chr(Decimal(trump))))
                break

            elif response == '115' and trump == '':
                trump = '115'
                namer = round_seq[i]
                previous.append('o')
                print("{0} orders {1} to be trump".format(str(round_seq[i].name), chr(Decimal(trump))))
                break

            else:
                previous.append('p')

    # playing round
    hand_loser = []
    hand_winner = []
    skip_trigger = False

    round_winner = -1
    for hand_round in range(5):
        lead_suit = ""
        first = False
        board = [0] * 4
        if hand_round > 0:
            round_seq = sequence(round_winner, round_seq)
        for player in range(len(round_seq)):
            if round_seq[player].playing:
                response = round_seq[player].play_card(board, trump)

                if response == 0:
                    print("{0} played a card that they already played, automatic lose".format(round_seq[player].name))
                    hand_loser.append(round_seq[player])
                    round_seq[player].valuation = -3
                    skip_trigger = True


                else:
                    if not first:
                        lead_suit = str(response)[-3:]
                        print("\n{0} leads with {1}".format(round_seq[player].name, convert_card_2_char(str(response))))
                        board[player] = response
                        first = True
                    else:
                        print("{0} plays {1}".format(round_seq[player].name, convert_card_2_char(str(response))))
                    board[player] = response
        print("board state is {0} after round {1}".format(board, hand_round))

        if skip_trigger:
            hand_winner = [x for x in players if x not in hand_loser]
            for player in hand_winner:
                player.valuation = 4  # fixme switch to zero after rewards system works
            return hand_winner, hand_loser

        idx = d.winner(board, trump)
        round_seq[idx].tricks += 1
        round_seq[idx].partner.tricks += 1
        round_winner = idx
        print("\n{0} won the hand\n".format(round_seq[round_winner].name))



        print("\n")

        """
        there needs to be a decision here for if the model gets rewarded for getting tricks, or for just winning
        the tricks should probably mean that its just a mlp while the winning should be a rnn that updates after 5 time 
        steps which represent the 5 rounds in a hand
        """
        #winners = (round_seq[idx], round_seq[idx].partner)
        #losers = (x for x in players if x not in winners)
        # reward round


    for player in range(2):
        if round_seq[player].tricks >= 3:
            hand_winner.append(round_seq[player])
            hand_winner.append(round_seq[player].partner)

            if round_seq[player].tricks >= 3 and round_seq[player].tricks < 5:
                if round_seq[player].partner.playing == True:
                    round_seq[player].partner.valuation = 1
                if round_seq[player].playing == True:
                    round_seq[player].valuation = 1

            if round_seq[player].tricks == 5:
                if round_seq[player].partner.playing == True:
                    round_seq[player].partner.valuation = 2
                    if round_seq[player].playing == False:
                        round_seq[player].partner.valuation = 4

                if round_seq[player].playing == True:
                    round_seq[player].valuation = 2
                    if round_seq[player].partner.playing == False:
                        round_seq[player].valuation = 4

    hand_loser = [x for x in players if x not in hand_winner]
    for player in hand_loser:
        player.valuation = -1

    if namer in hand_loser:
        print("{0} and {1} got bumped by the other team".format(namer.name, namer.partner.name))
        namer.valuation = -2
        namer.partner.valuation = -2
        for player in hand_winner:
            player.valuation = 2
            player.partner.valuation = 2

    else:
        print("{0} and {1} are the winners".format(namer.name, namer.partner.name))

    # reward hand
    return hand_winner, hand_loser


class Deck(object):

    def __init__(self, players, mode, batch_size):

        if mode == 'test':
            self.deck = []
        elif mode == 'train':
            self.deck = [[]]*batch_size

        self.bs = batch_size
        self.mode = mode
        self.players = players
        self.alts = {'115': '099', '099': '115', '104': '100', '100': '104'}

        # tranks = 'akqt9'
        self.tranks = ['097', '107', '113', '116', '057']
        # cranks = 'akqjt9'
        self.cranks = ['097', '107', '113', '106', '116', '057']

    def winner(self, cards, trump):
        if '106' + trump in cards:
            return cards.index('106' + trump)
        if '106' + self.alts[trump] in cards:
            return cards.index('106' + self.alts[trump])
        for r in self.tranks:
            if r + trump in cards:
                return cards.index(r + trump)

        # Fixme once the zero punishment reinforcement has been added
        for i in range(4):
            if cards[i] != 0:
                lead = cards[i][-3:]
                break

        for r in self.cranks:
            if r + lead in cards:
                return cards.index(r + lead)
        return random.randint(0,3)

    def make_deck(self):
        if self.mode == 'test':
            for s in suits:
                for v in values:
                    self.deck.append(v + s)
            random.shuffle(self.deck)
        else:
            for x in range(self.bs):
                for s in suits:
                    for v in values:
                        self.deck[x].append(v + s)
                random.shuffle(self.deck[x])


    def deal(self):

        cards = []
        for i in range(20):
            x= self.deck.pop(0)
            cards.append(x)
        self.players[0].hand.append(cards.pop(0))
        self.players[0].hand.append(cards.pop(0))
        self.players[0].hand.append(cards.pop(0))

        self.players[1].hand.append(cards.pop(0))
        self.players[1].hand.append(cards.pop(0))

        self.players[2].hand.append(cards.pop(0))
        self.players[2].hand.append(cards.pop(0))
        self.players[2].hand.append(cards.pop(0))

        self.players[3].hand.append(cards.pop(0))
        self.players[3].hand.append(cards.pop(0))

        self.players[0].hand.append(cards.pop(0))
        self.players[0].hand.append(cards.pop(0))

        self.players[1].hand.append(cards.pop(0))
        self.players[1].hand.append(cards.pop(0))
        self.players[1].hand.append(cards.pop(0))

        self.players[2].hand.append(cards.pop(0))
        self.players[2].hand.append(cards.pop(0))

        self.players[3].hand.append(cards.pop(0))
        self.players[3].hand.append(cards.pop(0))
        self.players[3].hand.append(cards.pop(0))

        for player in self.players:
            player.add_suits()

        return self.deck

    def reshuffle(self):
        self.make_deck()


class Player(object):

    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.hand = []
        self.suits = []
        self.suits_played = []
        self.namer = False
        self.playing = True
        self.tricks = 0
        self.called = False
        self.ordered = False
        self.partner = None

        self.order_net = mo()
        self.call_net = mc()
        self.strat = gameplan()
        self.hidden = self.strat.init_hidden(1)
        self.dis = dis()

        self.optim_order = optim.Adam(self.order_net.parameters(), lr=1e-2)
        self.optim_call = optim.Adam(self.call_net.parameters(), lr=1e-2)
        self.optim_strat = optim.Adam(self.strat.parameters(), lr=1e-2)
        self.optim_dis = optim.Adam(self.dis.parameters(), lr=1e-2)

        self.var_order = Variable(requires_grad=True)
        self.var_call = Variable(requires_grad=True)
        self.var_strat = Variable(requires_grad=True)
        self.var_dis = Variable(requires_grad=True)

        self.valuation = 0


    def order(self, shown):
        shown_tensor = tensor([float(shown)]).view(1,1)
        hand_tensor = tensor([float(i) for i in self.hand]).view(1,5)
        state = torch.cat([shown_tensor, hand_tensor], 1)
        self.var_order.data = tensor(state)
        choice_array = self.order_net.forward(self.var_order)
        val, idx = torch.max(choice_array, 1)

        if idx.data.view(1,1)[0][0] > 0:
            self.ordered = True
        return idx.data.view(1,1)[0][0]

    def call(self):
        self.var_call.data = tensor([float(i) for i in self.hand]).view(1, 5)
        choice_array = self.call_net.forward(self.var_call)
        val, idx = torch.max(choice_array, 1)
        if idx.data.view(1,1)[0][0] > 0:
            self.called = True
        return suits[idx.data.view(1,1)[0][0]-1]

    def play_card(self, outside_state, trump):
        state = torch.cat([tensor([float(i) for i in outside_state]).view(1,4),
                           tensor([float(i) for i in self.hand]).view(1,5), tensor([float(trump)]).view(1,1)], 1)
        self.var_strat.data = state
        card_to_play, self.hidden = self.strat.forward(self.var_strat, self.hidden)
        val, idx = torch.max(card_to_play.view(1,5), 1)
        idx = idx.data.view(1,1)[0][0]
        outgoing_card = self.hand[idx]
        self.hand[idx] = 0
        return outgoing_card

    def discard(self, pickup, trump):
        state = torch.cat([tensor([float(pickup)]).view(1,1), tensor([float(i) for i in self.hand]).view(1,5),
                           tensor([float(trump)]).view(1,1)], 1)
        self.var_dis.data = tensor(state)
        choice_array = self.dis.forward(self.var_dis)
        val, idx = torch.max(choice_array, 1)
        discarded = self.hand[idx.data.view(1, 1)[0][0]]
        self.hand[idx.data.view(1,1)[0][0]] = pickup
        return discarded

    def add_suits(self):
        for card in self.hand:
            if card[1] not in self.suits:
                self.suits.append(card[1])

    @staticmethod
    def convert_dec(str):
        val = float(str[0])
        suit = int(str[1])
        print(val, suit)




