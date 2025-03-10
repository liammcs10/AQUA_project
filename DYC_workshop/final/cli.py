# Command line interface for the final challenge.
import argparse


def command_line():
    parser = argparse.ArgumentParser(description = 'Example of a command line interface in python',
                                    epilog = 'Author: Liam McSweeney, 2024, UoS')

    parser.add_argument('--config', help='Configuration file name', type = str)    # optional argument
    parser.add_argument('--timestamp', help='Value of timestamp', type = float)
    parser.add_argument('--save', help='Whether or not to save output', action = 'store_true')

    # Can enforce the type of the value expected, or options from a list. 

    # Tell the command line to run the arguments passed by the user
    args = parser.parse_args()

    return args
