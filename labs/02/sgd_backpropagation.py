#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.utils.tensorboard

import npfl138
npfl138.require_version("2425.2")
from npfl138 import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=20, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = torch.nn.Parameter(
            torch.randn(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer) * 0.1,
            requires_grad=True,  # This is the default.
        )
        self._b1 = torch.nn.Parameter(torch.zeros(args.hidden_layer))

        self._W2 = torch.nn.Parameter(
            torch.randn(args.hidden_layer, MNIST.LABELS) * 0.1,
            requires_grad=True,  # This is the default.
        )
        self._b2 = torch.nn.Parameter(torch.zeros(MNIST.LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.to(torch.float32)
        x = x / 255
        x = x.reshape((inputs.shape[0], -1))

        x = x @ self._W1 + self._b1
        x = torch.tanh(x)
        x = x @ self._W2 + self._b2

        return x

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        self.train()
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.C, MNIST.H, MNIST.W]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            images = batch["images"].to(self._W1.device)
            labels = batch["labels"].to(self._W1.device).to(torch.int64)

            logits = self(images)

            probabilities = torch.softmax(logits, dim=1)

            labels_onehot = torch.nn.functional.one_hot(labels, MNIST.LABELS)
            loss = torch.mean(- torch.sum(labels_onehot * torch.log(probabilities), dim=1))

            # We create a list of all parameters. Note that a `torch.nn.Module` automatically
            # tracks owned parameters, so we could also use `list(self.parameters())`.
            parameters = [self._W1, self._b1, self._W2, self._b2]

            self.zero_grad()
            loss.backward()

            gradients = [parameter.grad for parameter in parameters]
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter.sub_(self._args.learning_rate * gradient)

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self.eval()
        with torch.no_grad():
            # Compute the accuracy of the model prediction
            correct = 0
            for batch in dataset.batches(self._args.batch_size):
                images = batch["images"].to(self._W1.device)
                labels = batch["labels"].numpy(force=True).astype(np.int64)

                logits = self(images).numpy(force=True)

                predictions = np.argmax(logits, axis=1)
                correct += np.sum(predictions == labels)

        return correct / len(dataset)


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load raw data.
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    # Try using an accelerator if available.
    if torch.cuda.is_available():
        model = model.to(device="cuda")
    elif torch.mps.is_available():
        model = model.to(device="mps")
    elif torch.xpu.is_available():
        model = model.to(device="xpu")

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy), flush=True)
        writer.add_scalar("dev/accuracy", 100 * dev_accuracy, epoch + 1)

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
