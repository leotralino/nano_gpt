import logging
import sys


class CustomFormatter(logging.Formatter):
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    RED = "\033[31m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno >= logging.ERROR:
            level_color = self.RED
        elif record.levelno >= logging.WARNING:
            level_color = self.YELLOW
        else:
            level_color = self.WHITE

        fmt = (
            f"{self.MAGENTA}%(asctime)s{self.RESET} | "
            f"{level_color}%(levelname)s{self.RESET} | "
            f"{self.GREEN}%(filename)s{self.RESET}:"
            f"{self.BLUE}%(funcName)s{self.RESET}:"
            f"{self.YELLOW}L%(lineno)d{self.RESET} | - "
            f"{self.WHITE}%(message)s{self.RESET}"
        )

        self._style._fmt = fmt
        return super().format(record)


def setup_logging():
    """Configures the root logger with our custom colorful style."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logging is called twice
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = CustomFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
