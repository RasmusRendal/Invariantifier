"""File which contains the options for the program"""
from caching import Cache
import argparse
import sys


class Options:
    """options class for storing various options for running the script"""

    def post_init(self):
        """post-initialization code"""
        self.enlarge = self.step % 90 != 0

    def __init__(self):
        self.enlarge = False
        self.debug = False
        self.step = 20
        self.combine = False
        self.examples = 100
        self.use_sum = False
        self.cache_dir = "/tmp/P5/cache/"
        self.cache = Cache(self.cache_dir)
        self.convlayers = 0
        self.serial = False
        self.representatives = False
        self.samples = 100

    def parse_args(self, xtest_len):
        parser = argparse.ArgumentParser(description='Identify some numbers')
        parser.add_argument('samples', type=int, default=60000,
                            nargs='?', help='How many samples to check')
        parser.add_argument(
            '--examples',
            dest='examples',
            type=int,
            default=20,
            nargs='?',
            help='How many training examples to refer to')
        parser.add_argument(
            '--step',
            dest='step',
            type=int,
            default=20,
            nargs='?',
            help='The step in the rotation')
        parser.add_argument(
            '--layers',
            dest='layers',
            type=int,
            default=0,
            nargs='?',
            help='How many convolutional layers to use')
        parser.add_argument('--debug', dest='debug', action='store_true',
                            help='Saves cnn images to tmp folder')
        parser.add_argument('--combine', dest='combine', action='store_true',
                            help='Combines patches before comparing')
        parser.add_argument(
            '--sum',
            dest='use_sum',
            action='store_true',
            help="Sum errors for each rotation when finding best rotation")
        parser.add_argument('--rep', dest='use_rep', action='store_true',
                            help="Pick representative training set examples")
        args = parser.parse_args()
        self.debug = args.debug
        self.step = args.step
        self.combine = args.combine
        self.examples = args.examples
        self.convlayers = args.layers
        if args.samples > xtest_len:
            args.samples = xtest_len
            print(
                "Automatically reduced the amount of samples to " +
                str(xtest_len))
        self.samples = args.samples
        if 360 % args.step != 0:
            print("Step must be divisible by 360")
            sys.exit(1)
        self.use_sum = args.use_sum
        self.representatives = args.use_rep
        self.post_init()
