
from configparser import ConfigParser

def read_conf(file):
    """
    Reads the input config file and returns a dictionary of values.
    
    """
    parser = ConfigParser()

    parser.read(file)

    config_dict = {}
    for section in parser.sections():
        config_dict[section] = dict(parser[section])

    return config_dict



def write_conf(configuration, filename):
    """
    Write the configuration dictionary to a file

    Parameters
    ----------


    Return
    ----------


    """
    # Define the config object
    config = ConfigParser()

    # fill up the config object
    for section in configuration:
        config[section] = configuration[section]

    # save the config object to file
    with open(filename, 'w') as filetosave:
        config.write(filetosave)