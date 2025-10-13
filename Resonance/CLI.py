# Command line interface for the Resonance parameter search. 
import argparse

def command_line():
    parser = argparse.ArgumentParser(description = 'This file generates the resonance properties of neurons over autaptic parameter space.',
                                    epilog = 'Author: L. McSweeney, 2025, UoS')

    parser.add_argument('--sim', help='Whether to run simulation', action = 'store_true')
    parser.add_argument('--config', help='Configuration file for the simulation', type = str)
    parser.add_argument('--outfile', help='output filename as .pickle. Also used for plotting.', type = str)
    #parser.add_argument('-o', '--out', help='output file name', type = str)
    parser.add_argument('--save', help='whether or not to save the output file', action = 'store_true')
    # Can enforce the type of the value expected, or options from a list. 

    # Tell the command line to run the arguments passed by the user
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    print(command_line())