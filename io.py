import pickle
from pathlib import Path

def save_cells(cell, paths, fn='cells.pickle'):
    save_path = Path(paths['s2p']).parent
    with open(Path(save_path, fn), 'wb') as file:
        pickle.dump(cell, file)
    print(f"Successfully saved cells at: {Path(save_path, fn)}")

def load_cells(paths, fn='cells.pickle'):
    load_path = Path(paths['s2p']).parent
    with open(Path(load_path, fn), 'rb') as file:
        cells = pickle.load(file)
    print(f"Successfull loaded cells from {Path(load_path, fn)}")
    return cells