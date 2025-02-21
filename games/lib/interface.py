# This is the file that contains functions used to configure the League interface

import platform
import os

def init_colors():
    """Initialize color support for Windows PowerShell"""
    if platform.system() == 'Windows':
        # Enable ANSI colors in Windows PowerShell
        os.system('') # This triggers VT100 mode

def colored(text: str, color: str) -> str:
    # Initialize color support
    init_colors()
    
    prefixes = {
        'red': 'Error: ',
        'yellow': 'Warning: ',
        'blue': 'Debug: ',
        'green': 'Success: ',
        'orange': 'Scored: '
    }
    
    # Using standard ANSI codes that work in both environments
    color_codes = {
        'red': '\033[31m',      # Using standard red instead of bright red
        'orange': '\033[33m',   # Using yellow as fallback for orange
        'yellow': '\033[33m',
        'green': '\033[32m',
        'blue': '\033[34m'
    }
    
    prefix = prefixes.get(color.lower(), '')
    color_code = color_codes.get(color.lower(), '')
    return f"{color_code}{prefix}{text}\033[0m"

if __name__ == "__main__":
    # Test the colors
    test_messages = [
        ("This is a test message", "green"),
        ("This is a test message", "red"),
        ("This is a test message", "yellow"),
        ("This is a test message", "blue"),
        ("This is a test message", "orange"),
        ("This is a test message", "purple")
    ]
    
    for message, color in test_messages:
        print(colored(message, color))