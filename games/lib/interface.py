# This is the file that contains functions used to configure the League interface


# Colored output text
def colored(text: str, color: str) -> str:
    prefixes = {
        'red': 'Error: ',
        'yellow': 'Warning: ',
        'blue': 'Debug: ',
        'green': 'Success: ',  # Common for success/completion messages
        'orange': 'Scored: '  # Good for notable but not warning messages
    }
    # Define color codes separately
    color_codes = {
        'red': '\033[91m',
        'orange': '\033[38;5;208m',
        'yellow': '\033[93m',
        'green': '\033[92m',
        'blue': '\033[94m'
    }
    prefix = prefixes.get(color.lower(), '')
    color_code = color_codes.get(color.lower(), '')
    return f"{color_code}{prefix}{text}\033[0m"

if __name__ == "__main__":
    print(colored("This is a test message", "green"))
    print(colored("This is a test message", "red"))
    print(colored("This is a test message", "yellow"))
    print(colored("This is a test message", "blue"))
    print(colored("This is a test message", "orange"))
    print(colored("This is a test message", "purple"))