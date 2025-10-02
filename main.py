from lib.game.game_engine import GameEngine
from lib.core.console import LogLevel

result = GameEngine.launch_from_files(
    config_main="example_config.yaml",
    extra_defs="config/game_config.yml",
    red_strategy="example_atk",
    blue_strategy="example_def",
    log_name="example_result",
    set_level=LogLevel.WARNING,
)
print(result)
