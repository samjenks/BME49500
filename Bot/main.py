import random
from Bot.packages.Engine import Player
import numpy as np
import torch
from torch.autograd import Variable
from torch import FloatTensor as tensor
from Bot.packages.Engine import play_hand_test
from Bot.packages.Engine import random_partners

BATCH_SIZE = 64

rewards = [[], [], [], []]


def reward(w, l):
    for player in w:
        rewards[player.id].append(player.valuation)
    for player in l:
        rewards[player.id].append(player.valuation)


def update(players):
    for player in players:
        unit = tensor(rewards[player.id])
        #print("unit", unit)
        unit = (unit - unit.mean()) / (unit.std() + np.finfo(np.float32).eps)
        #print("unit", unit)

        player.optim_call.zero_grad()
        player.optim_dis.zero_grad()
        player.optim_order.zero_grad()
        player.optim_strat.zero_grad()

        player.var_dis.data = tensor([float(unit.sum())])
        player.var_strat.data = tensor([float(unit.sum())])
        player.var_call.data = tensor([float(unit.sum())])
        player.var_order.data = tensor([float(unit.sum())])

        #policy_loss = Variable()
        #print(policy_loss)
        #policy_loss.backward()
        player.var_dis.backward()
        player.var_strat.backward()
        player.var_call.backward()
        player.var_order.backward()


        player.optim_call.step()
        player.optim_dis.step()
        player.optim_order.step()
        player.optim_strat.step()



def play_round():

    # training code
    players = [[Player("bot 1", 0), Player("bot 2", 1), Player("bot 3", 2), Player("bot 4", 3)]]*BATCH_SIZE
    dealer = [np.random.randint(0, 3, size=(1, BATCH_SIZE))]
    random_partners(players, 'train')

    #test code
    players = [Player("bot 1", 0), Player("bot 2", 1), Player("bot 3", 2), Player("bot 4", 3)]
    dealer = random.randint(0, 3)
    random_partners(players, 'test')
    (winners, losers) = play_hand_test(dealer, players)
    print("\n\nwinners:",winners, "\nlosers:",losers)
    return winners, losers


if __name__ == '__main__':


    for j in range(200):
        rewards = [[], [], [], []]
        for i in range(10):
            print("round: ",j, ":",i)
            w, l = play_round()
            reward(w, l)
            print(rewards, "\n\n")

        print("rewards going into update:", rewards)
        update(w+l)
