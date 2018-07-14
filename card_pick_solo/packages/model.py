from torch import nn
from torch.autograd import Variable



class ModelStrat_simple(nn.Module):
    def __init__(self, state_space, action_space):
        super(ModelStrat_simple, self).__init__()
        self.input = nn.Linear(state_space, 18)
        self.h1 = nn.Linear(18, 36)
        self.h2 = nn.Linear(36, 64)
        self.h3 = nn.Linear(64, 64)
        self.h4 = nn.Linear(64, 64)
        self.h5 = nn.Linear(64, 32)
        self.h6 = nn.Linear(32, 16)
        self.policy = nn.Linear(16, action_space)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(10)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        y = self.ReLu(self.input(x))
        y = self.ReLu(self.h1(y))
        x = self.ReLu(self.h2(y))

        y = self.ReLu(self.h3(x))
        y = self.ReLu(self.h4(y))
        x = x + y
        y = self.ReLu(self.h5(x))
        x = self.ReLu(self.h6(y))

        return self.policy(x)

class ModelStrat_complex(nn.Module):
    def __init__(self, state_space, action_space):
        super(ModelStrat_complex, self).__init__()
        self.rnn1 = nn.GRU(input_size=state_space,
                           hidden_size=248,
                           num_layers=1)
        self.dense1 = nn.Linear(248, 124)
        self.dense2 = nn.Linear(124, 60)
        self.dense3 = nn.Linear(60, 10)
        self.policy = nn.Linear(10, action_space)

    def forward(self, x, hidden):
        x = x.view(1, 1, 32)
        x, hidden = self.rnn1(x, hidden)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.policy(x)
        return x.view(1,5), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, 248).zero_())
