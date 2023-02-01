import pandas as pd
import numpy as np
import itertools
from holofun.s2p import Suite2pData
import copy

def create_cells(data:np.ndarray, s2p:Suite2pData):
    stat = s2p.get_stat_iscell()
    data = np.transpose(data, (1, 0 ,2)) # put cells first
    cells = list()
    for ii in range(data.shape[0]):
        cells.append(Cell(np.squeeze(data[ii, :, :]), stat[ii], s2p.fr))
    return cells

class Cell:
    def __init__(self, neural_data:np.ndarray, s2p_data:dict, frame_rate:int):
        self.s2p_data = s2p_data # this should be a dict, from s2p's stat.npy
        self.neural_data = neural_data
        self.labels = pd.DataFrame()
        self.outputs = pd.DataFrame()
        self.time = np.arange(neural_data.shape[1])/frame_rate
        self.response_win = []

    def add_label(self, input, trim=False):
        input = self._check_inputs(input)
        if trim:
            input = input.iloc[range(self.neural_data.shape[0]), :]
        if input.shape[0] != self.neural_data.shape[0]:
            raise IndexError(f"Labels do not have the right number of trials ({self.neural_data.shape[0]})") # trials
        self.labels = self._add(input, self.labels)

    def add_output(self, input):
        input = self._check_inputs(input)
        self.outputs = self._add(input, self.outputs)

    def _add(self, input, df):

        for k, v in input.items():
            if k in df.keys():
                df[k] = v
            else:
                df = pd.concat([df, input], axis=1)
        return df

    def _check_inputs(self, input):
        if type(input) is not dict:
            raise TypeError(f"Input needs to be a dictionary, instead it was a {type(input)}")
        for k, v in input.items():
            if np.isscalar(v):
                input[k] = [v]
        return pd.DataFrame(input)

    def drop_trial(self, trials_to_drop):
        df = self.copy()
        # used to drop trials, eg not useful ones etc
        if trials_to_drop.dtype is np.dtype('bool'):
            trials_to_drop = np.asarray(np.where(trials_to_drop)[0])
        if not df.labels.empty: 
            df.labels.drop(index=trials_to_drop, inplace=True)
        df.neural_data = np.delete(df.neural_data, trials_to_drop, axis=0)
        return df

    def calculate_response_window(self):
        baseline_idx = (self.time > self.response_win[0]) & (self.time < self.response_win[1])
        response_idx = (self.time > self.response_win[2]) & (self.time < self.response_win[3])
        base = self.neural_data[:, baseline_idx].mean(axis=1)
        resp = self.neural_data[:, response_idx].mean(axis=1)
        return resp, base

    def average_over(self, select):
        # theoretically creating a dataframe?
        if type(select) is not list:
            select = [select]
        df = self.labels[select].copy()
        resp, base = self.calculate_response_window()
        df['data'] = resp - base
        m = (df.groupby([*select])
            .mean(numeric_only=True)
            .reset_index())
        e = (df.groupby([*select])
            .sem(numeric_only=True)
            .reset_index())
        m['error'] = e['data'] # mapping should be preserved
        return m

    # Getters and Setters
    def get_data_longform(self):
        out = pd.DataFrame(self.neural_data).melt()
        return

    @property
    def n_trials(self):
        return self.neural_data.shape[0]

    @property
    def n_frames(self):
        return self.neural_daters

    def set_response_window(self, input):
        self.response_win = input

    def copy(self):
        return copy.deepcopy(self)