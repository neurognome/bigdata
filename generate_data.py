# generate data

import numpy as np
from Cell import Cell
import matplotlib.pyplot as plt
n_trials = 500
trialwise = np.random.rand(100, 200, n_trials) # 100 cells, 200 frames, 50 trials
s2p_data = {'info': 3}

cells = list()
for ii in np.arange(trialwise.shape[0]):
    cells.append(Cell(neural_data=np.squeeze(trialwise[ii, :, :]), s2p_data=s2p_data))

for c in cells:
    label = np.random.randint(15, size=n_trials)
    c.add_label(label, 'test')

for c in cells:
    y, x = c.average_over('test')
    plt.plot(x, y)