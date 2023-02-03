from holofun.s2p import Suite2pData
import numpy as np
from classes import Cell

def create_cells(s2p:Suite2pData):
    stat = s2p.get_stat_iscell()
    cells = list()
    for ii in range(len(stat)):
        cells.append(Cell(stat[ii]))
    return cells

def add_dataset(cells, epoch, data, response_win, framerate):
    for idx, c in enumerate(cells):
        c.create_dataset(epoch, np.squeeze(data[:, idx, :]), response_win, framerate)