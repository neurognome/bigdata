import pandas as pd
import numpy as np
import copy

# shared methods 
def _add(input, df):
    for k, v in input.items():
        if k in df.keys():
            df[k] = v
        else:
            df = pd.concat([df, v], axis=1)
    return df

def _check_inputs(input):
    if type(input) is not dict:
        raise TypeError(f"Input needs to be a dictionary, instead it was a {type(input)}")
    for k, v in input.items():
        if np.isscalar(v):
            input[k] = [v]
    return pd.DataFrame(input)

class Recording:
    def __init__(self, data:np.ndarray, response_frames, framerate):
        self.time = np.arange(data.shape[1])/framerate
        self.data = pd.DataFrame(data)
        self.labels = pd.DataFrame()
        self.response_win = np.array(response_frames)/framerate

    def drop(self, trials_to_drop):
        df = self.copy()
        # used to drop trials, eg not useful ones etc
        if trials_to_drop.dtype is np.dtype('bool'):
            trials_to_drop = np.asarray(np.where(trials_to_drop)[0])
        if not df.labels.empty: 
            df.labels.drop(index=trials_to_drop, inplace=True)
        df.data.drop(index=trials_to_drop, inplace=True)
        return df
    
    def filter(self, trials_to_keep):
        df = self.copy()
        if trials_to_keep.dtype is np.dtype('bool'):
            trials_to_keep = np.asarray(np.where(trials_to_keep)[0]) 
        if not df.labels.empty:
            df.labels.filter(index=trials_to_keep, inplace=True)
        df.data.filter(index=trials_to_keep, inplace=True)
        return df

    def calculate_response_window(self):
        baseline_idx = (self.time > self.response_win[0]) & (self.time < self.response_win[1])
        response_idx = (self.time > self.response_win[2]) & (self.time < self.response_win[3])
        base = self.data.loc[:, baseline_idx].mean(axis=1)
        resp = self.data.loc[:, response_idx].mean(axis=1)
        return resp, base

    def add_label(self, input, trim=False):
        input = _check_inputs(input)
        if trim:
            input = input.iloc[range(self.n_trials), :]
        if input.shape[0] != self.n_trials:
            raise IndexError(f"Labels do not have the right number of trials ({self.data.shape[0]})") # trials
        self.labels = _add(input, self.labels)

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

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_trials(self):
        return self.data.shape[0]

    @property
    def n_frames(self):
        return self.data.shape[0]

class Cell:
    def __init__(self, s2p_data:dict):
        self.s2p_data = s2p_data # this should be a dict, from s2p's stat.npy
        self.recordings = dict() #pd.DataFrame()
        self.outputs = pd.DataFrame()

    def add_recording(self, epoch, data, response_win, framerate):
        self.recordings[epoch] = Recording(data, response_win, framerate)

    def add_output(self, input):
        input = _check_inputs(input)
        self.outputs = _add(input, self.outputs)