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

class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(input_size, output_size) * 0.1,
            requires_grad=False,
        )

        self.b = torch.nn.Parameter(
            torch.zeros(output_size),
            requires_grad=False
        )

    def forward(self, inputs):
        x = inputs @ self.W + self.b
        return x

    def grad(self, inputs, wrt_outputs):
        wrt_inputs = wrt_outputs @ self.W.T
        wrt_W = inputs[:,:,torch.newaxis] @ wrt_outputs[:, torch.newaxis, :]
        wrt_b = wrt_outputs

        return wrt_inputs, wrt_W, wrt_b

    def update(self, wrt_W, wrt_b, learning_rate):
        self.W -= learning_rate * torch.mean(wrt_W, axis=0)
        self.b -= learning_rate * torch.mean(wrt_b, axis=0)

class TanhLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        x = torch.tanh(inputs)
        return x
    
    def grad(self, inputs, wrt_outputs):
        wrt_inputs = wrt_outputs * (1 - torch.pow(torch.tanh(inputs), 2))
        return wrt_inputs

    def update(self):
        pass


class CrossEntropyLoss():
    def forward(self, logits, labels_onehot):
        probabilities = torch.softmax(logits, dim=1)
        loss = torch.mean(- torch.sum(labels_onehot * torch.log(probabilities), dim=1))
        return loss
    
    def grad(self, logits, labels_onehot):
        # grad =labels_onehot * (torch.softmax(logits, dim=1) - 1)
        grad = torch.softmax(logits, dim=1) - labels_onehot 
        return grad


class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self.Linear_1 = LinearLayer(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer)
        self.tanh_layer = TanhLayer()
        self.Linear_2 = LinearLayer(args.hidden_layer, MNIST.LABELS)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = inputs.to(torch.float32)
        x = x / 255
        x_in = x.reshape((inputs.shape[0], -1))

        x_after_L1 = self.Linear_1(x_in)
        x_tanh = self.tanh_layer(x_after_L1)
        x_after_L2 = self.Linear_2(x_tanh)

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tanh, and the input layer after reshaping.
        return x_in, x_after_L1, x_tanh, x_after_L2

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        self.train()
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.C, MNIST.H, MNIST.W]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            images = batch["images"].to(self.Linear_1.W.device)
            labels = batch["labels"].to(self.Linear_1.W.device).to(torch.int64)

            # TODO: Contrary to `sgd_backpropagation`, the goal here is to compute
            # the gradient manually, without calling `.backward()`. ReCodEx disables
            # PyTorch automatic differentiation during evaluation.
            #
            # Start by computing the input layer, the hidden layer, and the output layer
            # of the batch images using `self(...)`.
            labels_onehot = torch.nn.functional.one_hot(labels, MNIST.LABELS)

            x_in, x_after_L1, x_tanh, logits = self(images)

            loss = CrossEntropyLoss()
            
            loss_value = loss.forward(logits, labels_onehot)

            wrt_loss = loss.grad(logits, labels_onehot)

            wrt_L2, wrt_W, wrt_b = self.Linear_2.grad(x_tanh, wrt_loss)
            self.Linear_2.update(wrt_W, wrt_b, self._args.learning_rate)

            wrt_tanh = self.tanh_layer.grad(x_after_L1, wrt_L2)

            _, wrt_W, wrt_b = self.Linear_1.grad(x_in, wrt_tanh)
            self.Linear_1.update(wrt_W, wrt_b, self._args.learning_rate)
            
            # TODO: Compute the gradient of the loss with respect to all
            # parameters. The loss is computed as in `sgd_backpropagation`.
            #
            # During the gradient computation, you will need to compute
            # a batched version of a so-called outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`,
            # which you can achieve by using for example
            #   `A[:, :, torch.newaxis] * B[:, torch.newaxis, :]`
            # or with
            #   `torch.einsum("bi,bj->bij", A, B)`.

            # TODO: Perform the SGD update with learning rate `self._args.learning_rate`
            # for all model parameters.

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self.eval()
        with torch.no_grad():
            # Compute the accuracy of the model prediction
            correct = 0
            for batch in dataset.batches(self._args.batch_size):
                images = batch["images"].to(self.Linear_1.W.device)
                labels = batch["labels"].numpy(force=True).astype(np.int64)

                logits = self(images)[3].numpy(force=True)

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
