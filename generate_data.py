# generate data

import numpy as np
from Cell import Cell
import matplotlib.pyplot as plt
n_trials = 500
trialwise = np.random.rand(1, 200, n_trials) # 100 cells, 200 frames, 50 trials
s2p_data = {'info': 3}
# is_cell = np.load() # this is the vector from suite2p determining if it's a cell or not, we should add this..

cells = Cell(neural_data=np.squeeze(trialwise[0, :, :]), s2p_data=s2p_data, is_cell=is_cell[0], frame_rate=10)

label = np.random.randint(12, size=n_trials) # eg orientations
cells.add_label(label, 'test')
label = np.random.randint(3, size=n_trials) # eg sizes
cells.add_label(label, 'test2')
y, x = cells.average_over([0, 5, 5.1, 7], ['test', 'test2'])