# import the library
import argparse

parser = argparse.ArgumentParser(description = 'Example of a command line interface in python',
                                  epilog = 'Author: Liam McSweeney, 2024, UoS')

parser.add_argument('file', help='input data file to the program')             # positional
parser.add_argument('file2', help='Configuration file to the program')            # positional
parser.add_argument('-c', '--count', help='Number of counts per iteration')    # optional argument
parser.add_argument('-n', help='Number of iterations')
parser.add_argument('--max', help='Maximum population per iteration')

# Can enforce the type of the value expected, or options from a list. 

# Tell the command line to run the arguments passed by the user
args = parser.parse_args()

# All values passed are saved as a dictionary in args
print(args)
