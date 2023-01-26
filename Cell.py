import pandas as pd
import numpy as np
import itertools

class Cell:
    def __init__(self, neural_data:np.ndarray, s2p_data:dict, is_cell:bool, frame_rate:int):
        self.s2p_data = s2p_data # this should be a dict, from s2p's stat.npy
        self.neural_data = neural_data
        self.is_cell = is_cell
        self.labels = pd.DataFrame()
        self.outputs = pd.DataFrame()
        self.time = np.arange(neural_data.shape[0])/frame_rate

    def add_label(self, input, label_name=None):
        input, name = self._check_inputs(input, label_name)
        self.labels[name] = input

    def add_output(self, input, output_name=None):
        input, name = self._check_inputs(input, output_name)
        self.outputs[name] = input

    def _check_inputs(self, input, name):
        if type(input) is dict:
            name = list(input.keys())[0]
            input = list(input.values())[0]
        else:
            if name is None:
                raise TypeError(f"If input is not a dictionary, must include label names")
        if input.shape[0] != self.neural_data.shape[-1]:
            raise IndexError(f"Labels do not have the right number of trials ({self.neural_data.shape[-1]})") # trials
        return input, name

    def average_over(self, win, input=None):
        baseline_idx = (self.time > win[0]) & (self.time < win[1])
        response_idx = (self.time > win[2]) & (self.time < win[3])
        base = self.neural_data[baseline_idx, :]
        resp = self.neural_data[response_idx, :]
        if input is None: # grand average across conditions
            return np.mean(resp) - np.mean(base), np.arange(self.neural_data.shape[-1])

        # check if input is a list
        if type(input) is not list:
            input = [input]

        targets = list()
        u_targets = list()
        for i in input:
            # for all the inputs, get the proper values
            targets.append(self.labels[i])
            u_targets.append(sorted(self.labels[i].unique()))
        subtracted_response = np.zeros([len(x) for x in u_targets])
        # not the fastest way of doing this... can we think of something faster? idk
        for lin_idx, combo in enumerate(itertools.product(*u_targets)):
            # for each potential combination of targets (list product)
            select = np.zeros((len(targets), self.neural_data.shape[-1]))  # this is targets x trials
            for idx, val in enumerate(combo):
                # for each target, collect the trues for that value
                select[idx, :] = targets[idx].to_numpy() == val # should output trues or falses into the matrix
            sub_idx = np.unravel_index(lin_idx, subtracted_response.shape) # unravel from linear to subscript
            subtracted_response[sub_idx] = np.mean(resp[:, np.all(select, axis=0)], axis=(0, 1)) - np.mean(base[:, np.all(select, axis=0)], axis=(0, 1))
        return subtracted_response, u_targets