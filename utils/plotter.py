#!/usr/bin/env python3
import termplotlib as tpl
from os import get_terminal_size
from outcsv import get_csv
import argparse

parser = argparse.ArgumentParser(description='Make a graph from CSV')

parser.add_argument('file', nargs=1, help="The file to parse")
args = parser.parse_args()

x, y = get_csv(args.file[0])
fig = tpl.figure()
ts = get_terminal_size()
fig.plot(x, y[1], width=ts.columns, height=ts.lines)
fig.show()
