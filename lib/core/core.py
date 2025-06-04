import os
import sys
import inspect
import time
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
from typeguard import typechecked
from typing import Any, Iterable, Optional, Union, List, Tuple, Dict, Set

# Global log level setting - can be changed at runtime
class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4

# Default log level - show all messages
CURRENT_LOG_LEVEL = LogLevel.DEBUG

@typechecked
def set_log_level(level: LogLevel) -> None:
    """
    Set the minimum log level that will be displayed.
    
    Parameters:
        level (LogLevel): The minimum level to display.
    """
    global CURRENT_LOG_LEVEL
    CURRENT_LOG_LEVEL = level

@typechecked
def _should_log(level: LogLevel, debug: bool) -> bool:
    """Check if we should log based on level and debug flag."""
    return debug and level.value >= CURRENT_LOG_LEVEL.value

@typechecked
def error(text: str, debug: bool = True) -> None:
    """
    Print an error message with timestamp and caller info in red, if debug is True.
    
    Parameters:
        text (str): The error message to display.
        debug (bool): Flag indicating whether to print the message. Defaults to True.
    """
    if not _should_log(LogLevel.ERROR, debug):
        return
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    filename = os.path.basename(frame.f_code.co_filename)
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}][{filename}::{caller}] \033[31m✗ Error: {text}\033[0m")

@typechecked
def warning(text: str, debug: bool = True) -> None:
    """
    Print a warning message with timestamp and caller info in yellow, if debug is True.
    
    Parameters:
        text (str): The warning message to display.
        debug (bool): Flag indicating whether to print the message. Defaults to True.
    """
    if not _should_log(LogLevel.WARNING, debug):
        return
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    filename = os.path.basename(frame.f_code.co_filename)
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}][{filename}::{caller}] \033[33m⚠ Warning: {text}\033[0m")

@typechecked
def info(text: str, debug: bool = True) -> None:
    """
    Print an info message with timestamp and caller info in blue, if debug is True.
    
    Parameters:
        text (str): The info message to display.
        debug (bool): Flag indicating whether to print the message. Defaults to True.
    """
    if not _should_log(LogLevel.INFO, debug):
        return
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    filename = os.path.basename(frame.f_code.co_filename)
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}][{filename}::{caller}] \033[34mℹ Info: {text}\033[0m")

@typechecked
def success(text: str, debug: bool = True) -> None:
    """
    Print a success message with timestamp and caller info in green, if debug is True.
    
    Parameters:
        text (str): The success message to display.
        debug (bool): Flag indicating whether to print the message. Defaults to True.
    """
    if not _should_log(LogLevel.SUCCESS, debug):
        return
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    filename = os.path.basename(frame.f_code.co_filename)
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}][{filename}::{caller}] \033[32m✓ Success: {text}\033[0m")

@typechecked
def dbg(text: str, debug: bool = True) -> None:
    """
    Print a debug message with timestamp and caller info in cyan, if debug is True.
    
    Parameters:
        text (str): The debug message to display.
        debug (bool): Flag indicating whether to print the message. Defaults to True.
    """
    if not _should_log(LogLevel.DEBUG, debug):
        return
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    filename = os.path.basename(frame.f_code.co_filename)
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}][{filename}::{caller}] \033[36m🔍 Debug: {text}\033[0m")

@contextmanager
@typechecked
def timed_block(name: str, level: LogLevel = LogLevel.INFO, debug: bool = True):
    """
    Context manager that times a block of code and logs the execution time.
    
    Parameters:
        name (str): Name of the operation being timed.
        level (LogLevel): Log level to use for timing messages.
        debug (bool): Whether to print the messages.
    
    Usage:
        with timed_block("Data processing"):
            # code to time
    """
    start_time = time.time()
    if _should_log(level, debug):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name
        filename = os.path.basename(frame.f_code.co_filename)
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{current_time}][{filename}::{caller}] \033[35m⏱ Started: {name}\033[0m")
    
    try:
        yield
    finally:
        if _should_log(level, debug):
            elapsed = time.time() - start_time
            frame = inspect.currentframe().f_back
            caller = frame.f_code.co_name
            filename = os.path.basename(frame.f_code.co_filename)
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{current_time}][{filename}::{caller}] \033[35m⏱ Completed: {name} in {elapsed:.4f}s\033[0m")

@typechecked
def add_root_folder_to_sys_path() -> str:
    """
    Searches upward from this file's directory until a directory containing a 'lib' folder is found.
    Adds that directory (the project root) to the beginning of sys.path and returns its path.
    
    Returns:
        str: The absolute path of the root folder.
    
    Raises:
        Exception: If no directory containing 'lib' is found.
    """
    # Start at the directory of this file.
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Traverse upward until we find a directory that contains a 'lib' folder.
    while True:
        potential_lib = os.path.join(current_dir, "lib")
        if os.path.isdir(potential_lib):
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise Exception("Could not find a parent directory containing a 'lib' folder.")
        current_dir = parent_dir

if __name__ == "__main__":
    # Add the project root folder to sys.path
    root_folder = add_root_folder_to_sys_path()
    success(f"Project root folder added to sys.path: {root_folder}", debug=True)
    
    def test_logging() -> None:
        """Test all logging functions and features."""
        # Basic logging
        dbg("This is a debug message")
        info("This is an info message")
        success("This is a success message")
        warning("This is a warning message")
        error("This is an error message")
        
        # Test with debug set to False
        error("This message should not appear", debug=False)
        
        # Test log levels
        print("\nTesting log levels:")
        set_log_level(LogLevel.WARNING)  # Only show warnings and errors
        dbg("This debug message should not appear due to log level")
        info("This info message should not appear due to log level")
        warning("This warning message should appear")
        error("This error message should appear")
        
        # Reset log level for other tests
        set_log_level(LogLevel.DEBUG)
        
        # Test timed blocks
        print("\nTesting timed blocks:")
        with timed_block("Fast operation"):
            # Simulate some work
            sum([i*i for i in range(10000)])
        
        with timed_block("Slow operation"):
            # Simulate more intensive work
            time.sleep(0.1)
            sum([i*i for i in range(100000)])
    
    test_logging()