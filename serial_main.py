#!/usr/bin/env python3

from training_sample_finder import get_training_samples
from runner import check_some
from options import Options
from network import train_network, get_dataset, get_model, split_network
from time import perf_counter
from tqdm.auto import tqdm
import argparse
import cProfile
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_experiment(
        iterations,
        options,
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        only_convolutional,
        filename):
    with open(filename, 'w') as f:
        f.write("examples,time,accuracy\n")
        for i in tqdm(range(10, iterations), desc=filename):
            options.examples = i
            start_time = perf_counter()
            training_samples = get_training_samples(
                only_convolutional, x_train, y_train, options)
            res = check_some(
                only_convolutional,
                x_test,
                y_test,
                model,
                training_samples,
                options)
            time = perf_counter() - start_time

            f.write(str(i) + "," + str(time) + "," + str(res[1]) + "\n")
            f.flush()


def setup():
    options = Options()
    options.serial = True
    x_train, y_train, x_test, y_test = get_dataset(
        options)  # pylint: disable=unused-variable

    model = train_network(get_model(x_test), options)
    only_convolutional = split_network(
        model, options.convlayers)  # pylint: disable=unused-variable

    profiling_dir = 'profiling'
    experiment_dir = 'experiments'

    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument(
        '--profile',
        dest='profile',
        action='store_true',
        help='Profile the experiments')
    parser.add_argument(
        '--nobasic',
        dest='basic',
        action='store_false',
        help='Skip the basic experiment')
    parser.add_argument(
        '--norep',
        dest='rep',
        action='store_false',
        help='Skip the representatives experiment')
    parser.add_argument(
        '--noconv',
        dest='conv',
        action='store_false',
        help='Skip the convolution experiment')
    parser.add_argument(
        '--iterations',
        type=int,
        default=501,
        nargs='?',
        help='How many iterations to run')

    args = parser.parse_args()

    # TODO: Make this cleaner
    try:
        shutil.rmtree(experiment_dir)
    except OSError:
        pass
    os.mkdir(experiment_dir)

    try:
        shutil.rmtree(profiling_dir)
    except OSError:
        pass
    os.mkdir(profiling_dir)

    if args.basic:
        options = Options()
        options.serial = True
        options.convlayers = 0
        options.step = 20
        options.combine = True
        options.post_init()
        cmd_string = """run_experiment(args.iterations, options, x_train, y_train, x_test, y_test,
                        model, only_convolutional, experiment_dir + '/basic.csv')"""
        if args.profile:
            cProfile.run(cmd_string, profiling_dir + '/basic')
        else:
            exec(cmd_string)

    if args.rep:
        options = Options()
        options.serial = True
        options.convlayers = 0
        options.step = 20
        options.combine = True
        options.representatives = True
        options.post_init()
        cmd_string = """run_experiment(args.iterations, options, x_train, y_train, x_test, y_test,
                        model, only_convolutional, experiment_dir + '/representatives.csv')"""
        if args.profile:
            cProfile.run(cmd_string, profiling_dir + '/representatives')
        else:
            exec(cmd_string)

    if args.conv:
        options = Options()
        options.serial = True
        options.convlayers = 1
        options.step = 20
        options.combine = False
        options.representatives = False
        options.post_init()
        only_convolutional = split_network(model, options.convlayers)
        cmd_string = """run_experiment(args.iterations, options, x_train, y_train, x_test, y_test,
                        model, only_convolutional, experiment_dir + '/convolution1.csv')"""
        if args.profile:
            cProfile.run(cmd_string, profiling_dir + '/convolution1')
        else:
            exec(cmd_string)


if __name__ == "__main__":
    setup()
