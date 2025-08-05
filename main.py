import os
import pathlib
import traceback
import networkx as nx
import gamms

from game import run_game
import lib.core.logger as TLOG
from lib.core.console import *

set_log_level(LogLevel.SUCCESS)
    
current_path = pathlib.Path(__file__).resolve()
root_path = current_path.parent
# print("Current path:", current_path)
# print("Root path:", root_path)

import example_strategy as alpha_team
import example_strategy as beta_team

RESULT_PATH = os.path.join(root_path, "data/result")
alpha_name = alpha_team.__name__.split(".")[-1]
beta_name = beta_team.__name__.split(".")[-1]
print(alpha_name, beta_name)
logger = TLOG.Logger("test")
logger.set_metadata(
    {
        "alpha_team": alpha_name,
        "beta_team": beta_name,
    }
)

config_path = "AF3BF3A5B5_68fb24_r01.yml"  # Example path to a config file
# Example of using the updated runner with the new file structure
# You can now specify just the filename and it will be found automatically
alpha_payoff, beta_payoff, game_time, alpha_caps, beta_caps, alpha_killed, beta_killed = run_game(config_path, root_dir=str(root_path), alpha_strategy=alpha_team, beta_strategy=beta_team, logger=logger, visualization=True, debug=False)

# logger.write_to_file("test.json")
print("Final payoff (Alpha):", alpha_payoff)
print("Final payoff (Beta):", beta_payoff)
print("Game time:", game_time)
print("Alpha captures:", alpha_caps)
print("Beta captures:", beta_caps)
print("Alpha killed:", alpha_killed)
print("Beta killed:", beta_killed)