from lib.game.game_engine import GameEngine
from lib.core.console import LogLevel

result = GameEngine.launch_from_files(
    config_main="example/example_config.yml",
    extra_defs="config/game_config.yml",
    red_strategy="example.example_atk",
    blue_strategy="example.example_def",
    log_name=None,
    set_level=LogLevel.DEBUG,
    record_file=False,
    record_video=False,
    vis=True,
    # tiff_path="graphs/wp_rg16.tiff",  # Optional GeoTIFF overlay (Would not unless the matching GeoTIFF file is provided in the graphs folder)
)
print(result)