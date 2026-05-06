from lib.game.game_engine import GameEngine
from lib.core.console import LogLevel

result = GameEngine.launch_from_files(
    # config_main="config/test1.yml",
    config_main="config/archive/v6r3/game_configs_osm_c/R5B5F3-5/R5B5F3-5_run12.yml",
    # config_main="config/game_configs_osm_a/R5B5F3-5/R5B5F3-5_run13.yml",
    # config_main="config/wp_1v1.yml",
    # extra_defs = "config/game_config_wp.yml",
    extra_defs="config/game_config.yml",
    # red_strategy="example.example_atk",
    red_strategy="policies.attacker.gmu_atk_r3", 
    # red_strategy="policies.attacker.uncc_atk_F_r3",
    # red_strategy="policies.attacker.msu_atk_A2_r2",
    # red_strategy="policies.attacker.msu_atk_M2_r2",
    # red_strategy="policies.attacker.msu_atk_M3_r3",
    # red_strategy="policies.attacker.gatech_atk_r3",
    # blue_strategy="example.example_def",    
    # blue_strategy="policies.defender.gatech_def_r3",
    blue_strategy="policies.defender.gmu_def_r3",
    # blue_strategy="policies.defender.gmu_def_bhs_r2",
    # blue_strategy="policies.defender.msu_def_A2_r2",
    # blue_strategy="policies.defender.msu_def_M2_r2",
    # blue_strategy="policies.defender.msu_def_M3_r3",
    # blue_strategy="policies.defender.uncc_def_F_r3",
    # blue_strategy="policies.defender.uncc_def_A_r3",
    # blue_strategy="policies.defender.uncc_def_r2",
    # log_name="test_runx",
    log_name=None,
    set_level=LogLevel.DEBUG,
    record_file=False,  # Enable .ggr recording
    record_video=False,  # Enable real-time MP4 recording
    vis=True,  # Enable visualization (False for NO_VIS mode)
    # tiff_path="graphs/wp_rg1516.TIFF",  # Optional path to save per-timestep TIFF images (e.g., "output/tiff/")
)
print(result)
