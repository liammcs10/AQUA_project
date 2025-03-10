# Practice with configuration files

import configparser

parser = configparser.ConfigParser()

parser.read('test_config.ini')

# imported values are Strings by default

print(parser.sections())

options = parser.options('simulation')

items_in_simulation = parser.items('simulation')
print(items_in_simulation)

sim_dict = dict(parser['simulation'])

# Also getint(), getboolean(), getfloat() to get the desired data type.
time_step_with_get = parser.get('simulation', 'time_step')

# You can write a config file too
config = configparser.ConfigParser()
# equivalent to defining a dictionary!
config['simulation'] = {'time_step': 1.0, 'total_time': 200.0}
config['environment'] = {'gavity': 9.81, 'air_resistance': 0.02}
config['initial condition'] = {'velocity': 5.0, 'angle': 30.0, 'height': 0.5}

# Writing into a config file. 
with open('config_file_program.ini', 'w') as configfile:
    config.write(configfile)
