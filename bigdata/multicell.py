from holofun.s2p import Suite2pData
import numpy as np
from bigdata.classes import Cell

def create_cells(s2p:Suite2pData):
    stat = s2p.get_stat_iscell()
    cells = list()
    for ii in range(len(stat)):
        cells.append(Cell(stat[ii]))
    return cells

def add_recording(cells, epoch, data, response_win, framerate):
    for idx, c in enumerate(cells):
        c.add_recording(epoch, np.squeeze(data[:, idx, :]), response_win, framerate)

def add_output(cells, output_data, output_name=None):
    if len(cells) != len(output_data):
        raise Exception(f"Outputs (length: {len(output)}) and cells (length:{len(cells)}) are mismatched.")
    # output_data is either a list of dicts or a vector with a single "output name", format appropriately...
    if type(output_data[0]) is dict:
        output = output_data
    else:
        output = [{output_name:o} for o in output_data]
    for (c, o) in zip(cells, output):
        [c.add_output({k: v}) for (k, v) in o.items()] # if it's a dict

def get_output(cells, output_name):
    return np.array([c.outputs[output_name].values[0] for c in cells])

def get_trialwise_legacy(cells, epoch):
    out = np.array([c.recordings[epoch].data.values for c in cells])
    return np.transpose(out, (1, 0, 2))