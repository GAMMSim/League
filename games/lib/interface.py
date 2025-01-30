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
   prefix = prefixes.get(color.lower(), '')
   return f"{dict(red='\033[91m',orange='\033[38;5;208m',yellow='\033[93m',green='\033[92m',blue='\033[94m').get(color.lower(), '')}{prefix}{text}\033[0m"