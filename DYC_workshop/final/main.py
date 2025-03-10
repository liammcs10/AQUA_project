import cli
import conf
from simulation import Simulation

def main():

    args = cli.command_line()
    
    if args.config is None:
        print("No config file passed... exit")
        quit()
    
    config_dict = conf.read_conf(args.config)

    # if timestamp is passed, change 'Parameters->timestep' to timestamp.
    if args.timestamp:
        config_dict['time']['time_steps'] = args.timestamp

    print(config_dict)
    
    sim = Simulation(config_dict['Parameters']['l'], config_dict['Parameters']['m'], config_dict['Parameters']['h'])
    print(sim.get_total())

    if args.save:
        conf.write_conf(config_dict, 'final_conf.ini')



main()