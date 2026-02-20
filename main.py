from lib.game.game_engine import GameEngine
from lib.core.console import LogLevel

result = GameEngine.launch_from_files(
    config_main="config/test1.yml",
    # config_main="config/game_configs/R5B5F3-5/R5B5F3-5_run1.yml",
    extra_defs="config/game_config.yml",
    # red_strategy="example.example_atk",
    # red_strategy="policies.attacker.uncc_atk_F_r3",
    # red_strategy="policies.attacker.msu_atk_M 3_r3",
    # red_strategy="policies.attacker.gmu_atk_r3",
    red_strategy="policies.attacker.gatech_atk_r3",
    # blue_strategy="example.example_def",    
    # blue_strategy="policies.defender.uncc_def_r3",
    blue_strategy="policies.defender.gatech_def_r3",
    # blue_strategy="policies.defender.msu_def_M3_r3",
    # blue_strategy="policies.defender.gmu_def_bhs_r2",
    # blue_strategy="policies.defender.gmu_def_r3",
    # blue_strategy="policies.defender.uncc_def_F_r3",
    log_name=None,
    set_level=LogLevel.WARNING,
    record=False,  # Enable recording
    vis=True,  # Enable visualization (False for NO_VIS mode)
)
print(result)
