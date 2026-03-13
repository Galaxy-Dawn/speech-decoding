import logging
import sys
from contextlib import contextmanager
import time
import os
import psutil
import math

# Try to import colorama for colored console output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_INSTALLED = True
except ImportError:
    COLORAMA_INSTALLED = False

# Custom file handler, inherited from logging.FileHandler
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Flush immediately after each record

# Set up logging function
def setup_logging(log_file=None, level=logging.INFO):
    # Define new log level INFO_HIGH_LEVEL, more important than INFO
    INFO_HIGH_LEVEL = 25
    logging.addLevelName(INFO_HIGH_LEVEL, "INFO_HIGH")

    # Define a new logging method
    def info_high(self, message, *args, **kwargs):
        if self.isEnabledFor(INFO_HIGH_LEVEL):
            self._log(INFO_HIGH_LEVEL, message, args, **kwargs)

    logging.Logger.info_high = info_high  # Add info_high method to Logger class

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)  # Set global log level

    # Remove default handlers to avoid duplicate log output
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # If colorama is installed, use colored formatter
    if COLORAMA_INSTALLED:
        class ColorFormatter(logging.Formatter):
            LEVEL_COLOR = {
                logging.DEBUG: Style.DIM + Fore.WHITE,
                logging.INFO: Fore.GREEN,
                INFO_HIGH_LEVEL: Fore.CYAN,  # Color for INFO_HIGH
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Style.BRIGHT + Fore.RED
            }

            def format(self, record):
                log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
                formatter = logging.Formatter(log_fmt)
                level_color = self.LEVEL_COLOR.get(record.levelno, Fore.WHITE)
                # Add color to log level and message
                record.levelname = level_color + record.levelname + Style.RESET_ALL
                record.msg = level_color + record.getMessage() + Style.RESET_ALL
                return formatter.format(record)

        color_formatter = ColorFormatter()
        console_handler.setFormatter(color_formatter)  # Use colored formatter for console
    else:
        cprint("colorama not installed, using plain formatter")
        console_handler.setFormatter(formatter)  # Otherwise use plain formatter

    # If log_file is None or empty string, skip creating file handler
    if log_file and log_file != '':
        # Create custom file handler, output to file and ensure flush after each record
        file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Add file handler to logger
        logger.addHandler(file_handler)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger  # Return configured logger object


def cprint(text, color='WHITE', background=None, style="BOLD"):
    """
    Custom print function for colored text output
    :param text: Text to print
    :param color: Text foreground color (default is WHITE)
    :param background: Background color (optional)
    :param style: Style (e.g., BOLD, DIM, optional)
    """
    if not COLORAMA_INSTALLED:
        # If colorama is not installed, output directly
        print(text)
        return

    # Define foreground and background colors
    color_map = {
        'BLACK': Fore.BLACK, 'RED': Fore.RED, 'GREEN': Fore.GREEN, 'YELLOW': Fore.YELLOW,
        'BLUE': Fore.BLUE, 'MAGENTA': Fore.MAGENTA, 'CYAN': Fore.CYAN, 'WHITE': Fore.WHITE,
    }

    background_map = {
        'BLACK': Back.BLACK, 'RED': Back.RED, 'GREEN': Back.GREEN, 'YELLOW': Back.YELLOW,
        'BLUE': Back.BLUE, 'MAGENTA': Back.MAGENTA, 'CYAN': Back.CYAN, 'WHITE': Back.WHITE,
    }

    style_map = {
        'DIM': Style.DIM, 'NORMAL': Style.NORMAL, 'BOLD': Style.BRIGHT
    }

    # Get the corresponding values for the specified color, background color, and style
    text_color = color_map.get(color.upper(), Fore.WHITE)
    bg_color = background_map.get(background.upper(), '') if background else ''
    text_style = style_map.get(style.upper(), Style.NORMAL) if style else ''

    # Combine and output
    output = f"{text_color}{bg_color}{text_style}{text}{Style.RESET_ALL}"
    print(output)

@contextmanager
def tracking(block_name, logger):
    logger.info(f"Start executing code block: {block_name}")
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0 ** 30

    try:
        yield
        logger.info(f"Code block {block_name} executed successfully")
    except Exception as e:
        logger.error(f"Code block {block_name} execution failed, exception: {e}")
        raise
    finally:
        end_time = time.time()
        elapsed_time = end_time - t0
        m1 = p.memory_info().rss / 2.0 ** 30
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)

        logger.info(f"Code block {block_name} memory usage: [{m1:.1f}GB({sign}{delta:.1f}GB)]")
        logger.info(f"Code block {block_name} execution time: {elapsed_time:.4f} seconds")


# Test logging configuration
if __name__ == '__main__':
    logger = setup_logging(level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.info_high("This is an info_high message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")