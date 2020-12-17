#!/usr/bin/env python3
""" main module file
    run the script by './main.py' or 'python3 main.py'"""
import sys
import os
from src.runner import check_some
from src.options import Options
from src.training_sample_finder import get_training_samples
from src.network import train_network, get_dataset, get_model, split_network

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Initialize the options class.
    options = Options()
    # Parse arguments
    options.parse_args(10000)
    # Get the MNIST dataset
    x_train, y_train, x_test, y_test = get_dataset(options)
    # Gets the model and the conv part of model, the function also trains the
    # network, if there doesn't exist a saved network
    model = train_network(get_model(x_test, options), options)
    only_convolutional, _ = split_network(model, options.convlayers)
    model.summary()
    if only_convolutional is not None:
        only_convolutional.summary()
    training_samples = get_training_samples(
        only_convolutional, x_train, y_train, options)

    # Run the checksome function, which tests a chosen algorithm to find if it
    # can properly rotate pictures back
    check_some(
        only_convolutional,
        x_test,
        y_test,
        model,
        training_samples,
        options)
