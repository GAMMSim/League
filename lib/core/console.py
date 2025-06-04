from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme
from rich import box

dracula_theme = Theme(
    {
        "info": "bold #bd93f9",  # Cyan
        "warning": "bold #f1fa8c",  # Yellow
        "error": "bold #ff5555",  # Red
        "success": "bold #50fa7b",  # Green
        "attacker": "bold #ff5555",  # Pink
        "defender": "bold #8be9fd",  # Purple
        "header": "bold #f8f8f2 on #44475a",
        "setup": "bold #f8f8f2 on #6272a4",
        "value_good": "bold #50fa7b",
        "value_med": "bold #f1fa8c",
        "value_bad": "bold #ffb86c",
        "value_critical": "bold #ff5555",
        "title": "bold #f8f8f2 on #6272a4",
        "comment": "#6272a4",
        "foreground": "#f8f8f2",
        "background": "#282a36",
        "current_line": "#44475a",
        "selection": "#44475a",
        "orange": "bold #ffb86c",
        "pink": "bold #ff79c6",
    }
)
