#!/usr/bin/env python3
import argparse

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.3")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--models", default=2, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=43, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


class EnsambleModule(torch.nn.Module):
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    def forward(self, x):
        probs = torch.zeros((x.shape[0], MNIST.LABELS))
        for model in self.models:
            probs += torch.nn.functional.softmax(model(x), dim=1)

        return probs / len(self.models)


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the models.
    models: list[npfl138.TrainableModule] = []
    for model in range(args.models):
        models.append(npfl138.TrainableModule(torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(MNIST.H * MNIST.W * MNIST.C, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, MNIST.LABELS),
        )))

        models[-1].configure(
            optimizer=torch.optim.Adam(models[-1].parameters()),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        )

        print("Training model {}: ".format(model + 1), end="", flush=True)
        models[-1].fit(train, epochs=args.epochs, console=0)
        print("Done")

    individual_accuracies, ensemble_accuracies = [], []
    for i, model in enumerate(models):
        individual_accuracy = model.evaluate(dev)['test_accuracy']

        ensemble_model = npfl138.TrainableModule(
            EnsambleModule(models[:i+1])
        )
        ensemble_model.configure(
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)}
        )

        ensemble_accuracy = ensemble_model.evaluate(dev)['test_accuracy']

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(main_args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
