import os
import sys
import numpy as np

if __name__ == '__main__':
    module_path =    os.path.abspath(os.path.join('../'))
    if module_path not in sys.path:
        sys.path.append(module_path)

from utils.data import subsampling
from utils.file import load_from_json

# load configs
data_configs = load_from_json("../configs/demo/data-mnist.json")
output_root = "../../data/"

# load the full-sized benign samples
file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
X_bs = np.load(file)

# load true labels

file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(file)

subsamples, sublabels = subsampling(data=X_bs,
                                    labels=labels,
                                    num_classes=10,
                                    filepath=output_root,
                                    filename='mnist')
