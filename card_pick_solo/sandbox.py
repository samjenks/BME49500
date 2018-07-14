from card_pick_solo.packages.Euchre_env import Euchre
from torch import FloatTensor as tensor
import numpy as np

env = Euchre()

state = env.reset()

print(state)

print(env.step(1))

print(tensor(np.array(state).astype(np.float)))
