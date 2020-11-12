#!/usr/bin/env python3
import pstats
import argparse

parser = argparse.ArgumentParser(description='Print some profiling')

parser.add_argument('lines', metavar='N', type=int, nargs='?',
                    help='The amount of lines')
parser.add_argument('file', nargs=1, help="The file to parse")
args = parser.parse_args()
p = pstats.Stats(args.file[0])
p.sort_stats('cumulative').print_stats(args.lines)
