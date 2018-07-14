#!/usr/local/bin/python2.7

import sys, subprocess, random

players = []
partners = {}

suits = ['h','d','c','s']
values = ['9','t','j','q','k','a']
deck = []

def random_partners():
    random.shuffle(players)

def sequence(start):
    out = []
    for i in range(4):
        out.append(players[(start+i)%4])
    return out

def resequence(lst, offset):
    out = []
    for i in range(4):
        out.append(lst[(offset+i)%4])
    return out

def first_index_of_suit(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def play_game():
    global deck, players, partners
    dealer = random.randint(0,3)
    round_seq = []
    rounds = 0
    while players[0].score < 10 and players[1].score < 10:
        dealer = (dealer+1)%4
        print("\n-----------------------------")
        print("ROUND {0} START\n{1} is dealer".format(rounds, players[dealer]))
        print("TEAM 1 ({0}, {1}): {2} points".format(str(players[0]),str(players[2]),players[0].score))
        print("TEAM 2 ({0}, {1})): {2} points".format(str(players[1]),str(players[3]),players[1].score))
        for p in players:
            p.playing = True
            p.hand = []
            p.tricks = 0
        rounds += 1
        make_deck()
        round_seq = sequence(dealer+1)
        deal()
        trump = ''
        previous = []
        card = deck.pop()
        namer = -1
        print("\nORDERING Phase")
        for i in range(len(round_seq)):
            response = round_seq[i].trump(previous, card=card)
            print("{0}: {1}".format(str(round_seq[i]), response))
            if response == 'o' and trump == '':
                trump = card[-1]
                namer = i
                previous.append('o')
                print("{0} orders {1} to trump".format(str(round_seq[i]), trump))
            else:
                previous.append('p')
        if trump == '':
            previous = []
            print("\nNAMING Phase")
            for i in range(len(round_seq)):
                response = round_seq[i].trump(previous)
                print("{0}: {1}".format(str(round_seq[i]), response))
                if response in suits and trump == '':
                    trump = response
                    namer = i
                    previous.append(response)
                    print("{0} names {1} to trump\n\n".format(str(round_seq[i]), trump))
                else:
                    previous.append('p')
        else:
            print("\nDealer discarding")
            response = players[dealer].discard(card)
            if response in players[dealer].hand:
                del players[dealer].hand[players[dealer].hand.index(response)]
            else:
                del players[dealer].hand[0]
            players[dealer].hand.append(card)
        if trump:
            previous = []
            print("\nGOING ALONE Phase")
            for i in range(len(round_seq)):
                previous.append(round_seq[i].alone(trump, previous))
                print("{0}: {1}".format(str(round_seq[i]), previous[-1]))
            if previous[0] == 'y':
                round_seq[2].playing = False
            elif previous[2] == 'y':
                round_seq[0].playing = False
            if previous[1] == 'y':
                round_seq[3].playing = False
            elif previous[3] == 'y':
                round_seq[1].playing = False

            for _j in range(5):
                print("\nTRICK {0}".format(_j))
                previous = []
                lead = round_seq[0].turn(previous, trump)
                previous.append(lead)
                #print("\n{0} hand: {1}".format(str(round_seq[0]), ', '.join(round_seq[0].hand)))
                print("\n{0} leads {1}".format(str(round_seq[0]), lead))
                del round_seq[0].hand[round_seq[0].hand.index(lead)]
                for i in range(1,len(round_seq)):
                    #print("{0} hand: {1}".format(str(round_seq[i]), ', '.join(round_seq[i].hand)))
                    if round_seq[i].playing:
                        response = round_seq[i].turn(previous, trump)
                        if response in round_seq[i].hand:
                            if ((lead[-1] in "".join(round_seq[i].hand)) and (lead[-1] in response)) or not(lead[-1] in "".join(round_seq[i].hand)):
                                del round_seq[i].hand[round_seq[i].hand.index(response)]
                                previous.append(response)
                                print("{0} plays {1}".format(str(round_seq[i]), response))
                            else:
                                if lead[-1] in "".join(round_seq[i].hand):
                                    previous.append(round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])])
                                    print("{0} plays {1}".format(str(round_seq[i]), round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])]))
                                    del round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])]
                                else:
                                    previous.append(round_seq[i].hand[0])
                                    print("{0} plays {1}".format(str(round_seq[i]), round_seq[i].hand[0]))
                                    del round_seq[i].hand[0]
                        else:
                            if lead[-1] in "".join(round_seq[i].hand):
                                previous.append(round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])])
                                print("{0} plays {1}".format(str(round_seq[i]), round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])]))
                                del round_seq[i].hand[first_index_of_suit(round_seq[i].hand, lead[-1])]
                            else:
                                previous.append(round_seq[i].hand[0])
                                print("{0} plays {1}".format(str(round_seq[i]), round_seq[i].hand[0]))
                                del round_seq[i].hand[0]
                    else:
                        del round_seq[i].hand[0]
                        print("{0} is sitting out".format(str(round_seq[i])))
                        previous.append('p')
                w = winner(previous, trump)
                print("{0} wins the trick\n".format(str(round_seq[w])))
                round_seq[w].tricks += 1
                partners[round_seq[w]].tricks = round_seq[w].tricks
                for i in range(4):
                    round_seq[i].trick_data(previous, i, trump)
                round_seq = resequence(round_seq, w)
            if round_seq[namer].tricks < round_seq[(namer+1)%4].tricks:
                print("\nDefenders set the namers")
                round_seq[(namer+1)%4].score += 2
                round_seq[(namer+3)%4].score = round_seq[(namer+1)%4].score
            elif round_seq[namer].tricks == 5:
                if round_seq[(namer+2)%4].playing:
                    print("\nNamers won all tricks")
                    round_seq[namer].score += 2
                    round_seq[(namer+2)%4].score = round_seq[namer].score
                else:
                    print("\nNamers went alone and won all tricks")
                    round_seq[namer].score += 4
                    round_seq[(namer+2)%4].score = round_seq[namer].score
            else:
                print("\nNamers won {} tricks".format(round_seq[namer].tricks))
                round_seq[namer].score += 1
                round_seq[(namer+2)%4].score = round_seq[namer].score
            for p in players:
                p.tricks = 0
    if players[0].score >= 10:
        print("\n\n{0} and {1} won".format(str(players[0]), str(players[2])))
    else:
        print("\n\n{0} and {1} won".format(str(players[1]), str(players[3])))

def winner(cards, trump):
    if 'j'+trump in cards:
        return cards.index('j'+trump)
    alts = {'s':'c','c':'s','h':'d','d':'h'}
    if 'j'+alts[trump] in cards:
        return cards.index('j'+alts[trump])
    tranks = 'akqt9'
    for r in tranks:
        if r+trump in cards:
            return cards.index(r+trump)
    cranks = 'akqjt9'
    lead = cards[0][-1]
    for r in cranks:
        if r+lead in cards:
            return cards.index(r+lead)

def make_deck():
    global deck
    deck = []
    for s in suits:
        for v in values:
            deck.append(v+s)
    random.shuffle(deck)

def deal():
    global deck
    cards = []
    for i in range(20):
        cards.append(deck.pop(0))
    players[0].hand.append(cards.pop(0))
    players[0].hand.append(cards.pop(0))
    players[0].hand.append(cards.pop(0))

    players[1].hand.append(cards.pop(0))
    players[1].hand.append(cards.pop(0))

    players[2].hand.append(cards.pop(0))
    players[2].hand.append(cards.pop(0))
    players[2].hand.append(cards.pop(0))

    players[3].hand.append(cards.pop(0))
    players[3].hand.append(cards.pop(0))

    players[0].hand.append(cards.pop(0))
    players[0].hand.append(cards.pop(0))

    players[1].hand.append(cards.pop(0))
    players[1].hand.append(cards.pop(0))
    players[1].hand.append(cards.pop(0))

    players[2].hand.append(cards.pop(0))
    players[2].hand.append(cards.pop(0))

    players[3].hand.append(cards.pop(0))
    players[3].hand.append(cards.pop(0))
    players[3].hand.append(cards.pop(0))

class Player(object):
    def __init__(self, name):
        global players
        self.fname = name
        self.name = "{0} ({1})".format(name, len(players))
        self.hand = []
        self.score = 0
        self.namer = False
        self.playing = True
        self.tricks = 0

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return other.name == self.name

    def trump(self, prev, card=''):
        if card:
            instr = 'ordering\n'
            instr += '{}\n'.format(card)
            instr += '{}'.format(','.join(prev))
        else:
            instr = 'naming\n'
            instr += '{}'.format(','.join(prev))
        return self.communicate(instr)

    def discard(self, card):
        instr = 'discard\n'
        instr += '{}'.format(card)
        return self.communicate(instr)

    def alone(self, suit, prev):
        instr = 'alone\n'
        instr += '{}\n'.format(suit)
        instr += '{}'.format(','.join(prev))
        return self.communicate(instr)

    def turn(self, on_table, suit):
        instr = 'turn\n'
        instr += '{}\n'.format(suit)
        instr += '{}'.format(','.join(on_table))
        return self.communicate(instr)

    def trick_data(self, prev, n, trump):
        instr = 'trick\n'
        instr += '{}\n'.format(n)
        instr += '{}\n'.format(trump)
        instr += '{}'.format(','.join(prev))
        self.communicate(instr, out=False)

    def communicate(self, instring, out=True):
        instr = '{}\n'.format(','.join(self.hand))
        instr += '{}\n'.format(self.score)
        instr += '{}\n'.format(self.tricks)
        instr += instring
        #print(instr)
        bot = subprocess.Popen(['bots/'+self.fname], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if out:
            out = bot.communicate(input=instr)[0].strip()
            return out
        else:
            bot.communicate(input=instr)
            return

if __name__ == "__main__":
    for name in sys.argv[1:]:
        players.append(Player(name))
    if len(players) > 4 or len(players) in [1,3]:
        raise ValueError
    if len(players) == 2:
        players.append(Player(players[0].fname))
        players.append(Player(players[1].fname))
    else:
        random_partners()
    partners = {players[0]:players[2], players[2]:players[0], players[1]:players[3], players[3]:players[1]}
    play_game()
