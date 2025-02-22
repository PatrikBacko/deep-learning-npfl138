#!/usr/bin/env python3
import argparse

import numpy as np
from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    with open(args.data_path, "r") as data:
        data_counts = Counter([line.rstrip("\n") for line in data])
        data_sum = sum(data_counts.values())

    data_dist_dict = {line: count / data_sum for line, count in data_counts.items()}

    with open(args.model_path, "r") as model:
        model_probs = [tuple(line.rstrip("\n").split("\t")) for line in model]

    model_dist_dict = {line: float(prob) for line, prob in model_probs}

    model_dist = np.array([model_dist_dict.get(key, 0) for key in sorted(data_dist_dict.keys())])
    data_dist = np.array([data_dist_dict.get(key, 0) for key in sorted(data_dist_dict.keys())])
    print(model_dist)
    print(data_dist)


    entropy = - np.sum(data_dist * np.log(data_dist))
    crossentropy = - np.sum(data_dist * np.log(model_dist))
    kl_divergence = np.sum(data_dist * np.log(data_dist / model_dist))

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
