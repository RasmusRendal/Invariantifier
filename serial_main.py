#!/usr/bin/env python3

import argparse
import cProfile
import os
import shutil

from time import perf_counter

from tqdm.auto import tqdm

from src.training_sample_finder import get_training_samples
from src.runner import check_some
from src.options import Options
from src.network import train_network, get_dataset, get_model, split_network, get_last_conv_layer

profiling_dir = 'profiling'
experiment_dir = 'experiments'

def remove_caches():
    try:
        shutil.rmtree('/tmp/P5/')
    except OSError:
        pass


# pylint: disable=too-many-arguments
def run_experiment(iterations,
                   options,
                   x_train,
                   y_train,
                   x_test,
                   y_test,
                   model,
                   only_convolutional,
                   filename):
    with open(filename, 'w') as f:
        if options.accperclass:
            f.write("examples,time,accuracy,0,1,2,3,4,5,6,7,8,9\n")
        else:
            f.write("examples,time,accuracy\n")
        for i in tqdm(range(10, iterations, 20), desc=filename):
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

            if options.accperclass:
                f.write(str(i) + "," + str(time) + "," + str(res[10]))
                for j in range(len(res) - 1):
                    f.write("," + str(res[j]))
                f.write("\n")
            else:
                f.write(str(i) + "," + str(time) + "," + str(res[1]) + "\n")
            f.flush()


def basic(iterations, profile):
    options = Options()
    options.serial = True
    #options.accperclass = True
    options.convlayers = 0
    options.step = 20
    options.combine = True
    options.post_init()
    x_train, y_train, x_test, y_test = get_dataset(options)  # pylint: disable=unused-variable
    model = train_network(get_model(x_test, options), options)
    only_convolutional, _ = split_network(model, options.convlayers)  # pylint: disable=unused-variable
    cmd_string = """run_experiment(iterations, options, x_train, y_train, x_test, y_test,
                    model, only_convolutional, experiment_dir + '/basic.csv')"""
    if profile:
        cProfile.run(cmd_string, profiling_dir + '/basic')
    else:
        exec(cmd_string)

def rep(iterations, profile):
    options = Options()
    options.serial = True
    options.accperclass = True
    options.convlayers = 0
    options.step = 20
    options.combine = True
    options.representatives = True
    options.post_init()
    x_train, y_train, x_test, y_test = get_dataset(options)  # pylint: disable=unused-variable
    model = train_network(get_model(x_test, options), options)
    only_convolutional, _ = split_network(model, options.convlayers)  # pylint: disable=unused-variable
    cmd_string = """run_experiment(iterations, options, x_train, y_train, x_test, y_test,
                    model, only_convolutional, experiment_dir + '/representatives.csv')"""
    if profile:
        cProfile.run(cmd_string, profiling_dir + '/representatives')
    else:
        exec(cmd_string)


def conv(iterations, profile):
    options = Options()
    options.serial = True
    options.accperclass = True
    options.convlayers = 5
    options.step = 20
    options.combine = False
    options.representatives = False
    options.post_init()
    x_train, y_train, x_test, y_test = get_dataset(options)  # pylint: disable=unused-variable
    model = train_network(get_model(x_test, options), options)
    options.convlayers = get_last_conv_layer(model)+1
    only_convolutional, _ = split_network(model, options.convlayers)  # pylint: disable=unused-variable
    cmd_string = """run_experiment(iterations, options, x_train, y_train, x_test, y_test,
                    model, only_convolutional, experiment_dir + '/convolution1.csv')"""
    if profile:
        cProfile.run(cmd_string, profiling_dir + '/convolution1')
    else:
        exec(cmd_string)

def rot_first(iterations, profile):
    options = Options()
    options.serial = True
    options.convlayers = 5
    options.step = 20
    options.combine = False
    options.representatives = False
    options.rotate_first = True
    options.post_init()
    x_train, y_train, x_test, y_test = get_dataset(options)  # pylint: disable=unused-variable
    model = train_network(get_model(x_test, options), options)
    options.convlayers = get_last_conv_layer(model)+1
    only_convolutional, _ = split_network(model, options.convlayers)  # pylint: disable=unused-variable
    cmd_string = """run_experiment(iterations, options, x_train, y_train, x_test, y_test,
                    model, only_convolutional, experiment_dir + '/rotate_first.csv')"""
    if profile:
        cProfile.run(cmd_string, profiling_dir + '/rotate_first')
    else:
        exec(cmd_string)

def constraint(iterations, profile):
    delete = True
    if delete:
        remove_caches()
    options = Options()
    options.serial = True
    options.convlayers = 3
    options.step = 20
    options.combine = False
    options.representatives = False
    options.post_init()
    options.model_step = 20
    x_train, y_train, x_test, y_test = get_dataset(options)  # pylint: disable=unused-variable
    model = train_network(get_model(x_test, options), options)
    options.convlayers = get_last_conv_layer(model)+1
    only_convolutional, _ = split_network(model, options.convlayers)
    cmd_string = """run_experiment(iterations, options, x_train, y_train, x_test, y_test,
                    model, only_convolutional, experiment_dir + '/constraint.csv')"""
    if profile:
        cProfile.run(cmd_string, profiling_dir + '/constraint')
    else:
        exec(cmd_string)


def setup():
    options = Options()
    options.serial = True



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
        '--norot',
        dest='rot_first',
        action='store_false',
        help='Skip the rotate first experiment')
    parser.add_argument(
        '--nocon',
        dest='constraint',
        action='store_false',
        help='Skip the constraint experiment')
    parser.add_argument(
        '--iterations',
        type=int,
        default=501,
        nargs='?',
        help='How many iterations to run')

    args = parser.parse_args()

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

    remove_caches()

    if args.basic:
        basic(args.iterations, args.profile)

    if args.rep:
        rep(args.iterations, args.profile)

    if args.conv:
        conv(args.iterations, args.profile)

    if args.rot_first:
        rot_first(args.iterations, args.profile)

    if args.constraint:
        constraint(args.iterations, args.profile)


if __name__ == "__main__":
    setup()
