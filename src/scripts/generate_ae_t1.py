import os
import numpy as np

from datetime import time
from attacks.attack import generate
from utils.file import load_from_json
from utils.metrics import error_rate
from utils.model import load_lenet


def generate_ae(model, data, labels, attack_configs, save=False, output_dir=None):
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")
    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    for id in range(num_attacks):
        key = "configs{}".format(id)
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_configs.get(key)
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)

        # save the adversarial example
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            # save with a random name
            file = os.path.join(output_dir, "{}.npy".format(time.monotonic()))
            print("Save the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)

if __name__ == '__main__':
    # load configs
    model_configs = load_from_json("../configs/demo/model-mnist.json")
    attack_configs = load_from_json("../configs/demo/attack-zk-mnist.json")
    data_configs = load_from_json("../configs/demo/data-mnist.json")

    # load target model
    model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
    target = load_lenet(file=model_file, wrap=True)

    # load the benign samples

    data_file = os.path.join(data_configs.get('sub_dir'), data_configs.get('subsample_file'))
    data_bs = np.load(data_file)

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('sub_dir'), data_configs.get('sublabel_file'))
    labels = np.load(label_file)

    # number of samples???
    data_bs = data_bs[:10]
    labels = labels[:10]
    generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True , output_dir="../../results")