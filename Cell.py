import pandas as pd
import numpy as np


class Cell:
    def __init__(self, neural_data:np.ndarray, s2p_data:dict):
        self.s2p_data = s2p_data # this should be a dict, from s2p's stat.npy
        self.neural_data = neural_data
        self.labels = pd.DataFrame()
        self.outputs = pd.DataFrame()

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

    def average_over(self, input):
        # check if oculmn exists
        # if exists, group and return mean by that column of trialwise data right?
        if input is list:
            raise Exception(f"Multi-averaging not implemented yet... sorry")
        if input not in self.labels.columns:
            raise Exception(f"{input} does not exist in labels ({self.labels.columns})")
        target = self.labels[input]
        u_targets = sorted(target.unique())
        avg_response = np.zeros((1, len(u_targets)))
        for ct, u in enumerate(u_targets):
            idx = (target.to_numpy() == u)
            avg_response[:, ct] = np.mean(self.neural_data[:, idx])
        return np.squeeze(avg_response), u_targets

        


