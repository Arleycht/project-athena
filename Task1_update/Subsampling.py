import os
import sys
import numpy as np
from pathlib import Path

if __name__ == '__main__':

    project_path = Path().absolute().parent
    module_path =    os.path.abspath(os.path.join('../'))
    if module_path not in sys.path:
        sys.path.append(module_path)

from src.utils.data import subsampling
from src.utils.file import load_from_json

# load configs
data_configs = load_from_json(project_path.joinpath("Task1_update/configs/data-mnist.json"))
output_root = project_path.joinpath("Task1_update/data/subsample")

# load the full-sized benign samples
file = project_path.joinpath(project_path.joinpath("data/test_BS-mnist-clean.npy"))
X_bs = np.load(file)

# load true labels

file = project_path.joinpath(project_path.joinpath("data/test_Label-mnist-clean.npy"))
labels = np.load(file)

subsamples, sublabels = subsampling(data=X_bs,
                                    labels=labels,
                                    num_classes=10,
                                    ratio=0.05,
                                    filepath=output_root,
                                    filename='mnist')
