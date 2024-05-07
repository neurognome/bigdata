import pickle
from pathlib import Path

def save_cells(cell, s2p_path, fn='cells.pickle'):
    # save_path = s2p_path.parent
    save_path = Path(s2p_path).parent
    with open(Path(save_path, fn), 'wb') as file:
        pickle.dump(cell, file)
    print(f"Successfully saved cells at: {Path(save_path, fn)}")

def load_cells(s2p_path, fn='cells.pickle'):
    # load_path = Path(paths['s2p']).parent
    load_path = Path(s2p_path).parent
    with open(Path(load_path, fn), 'rb') as file:
        cells = pickle.load(file)
    print(f"Successfull loaded cells from {Path(load_path, fn)}")
    return cells