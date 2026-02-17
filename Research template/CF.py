"""
Converts the information in RS_config.ini into a dictionary.
"""

import configparser

def read_conf(file):
    """
    Reads and extracts the config file.

    Arguments
    ---------
    file :      str
                configuration file name

    Returns
    ---------
    conf_dict : str
                configuration dictionary   
    """

    # create parser object
    parser = configparser.ConfigParser()
    parser.optionxform = str

    # read the config file
    parser.read(file)

    config_dict = {}
    # convert each section of the config file to dictionary
    # append as a subdictionary to config_dict
    for section in parser.sections():
        config_dict[section] = dict(parser[section])
    
    return config_dict