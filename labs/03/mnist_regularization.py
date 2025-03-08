#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.3")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST(sizes={"train": 5_000})

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    model = torch.nn.Sequential()
    model.append(torch.nn.Flatten())
    model.append(torch.nn.Dropout(p=args.dropout))
    features = MNIST.C * MNIST.H * MNIST.W
    for hidden_layer in args.hidden_layers:
        model.append(torch.nn.Linear(features, features := hidden_layer))
        model.append(torch.nn.Dropout(p=args.dropout))
        model.append(torch.nn.ReLU())
    model.append(torch.nn.Linear(features, features := MNIST.LABELS))

    # Wrap the model in the TrainableModule.
    model = npfl138.TrainableModule(model)

    params = [
        {'params': [param for name, param in model.named_parameters() if 'bias' not in name]},
        {
            'params': [param for name, param in model.named_parameters() if 'bias' in name],
            'weight_decay': 0
        }
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model.configure(
        optimizer=optimizer,
        loss=loss,
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
