import logging
import sys

from app.config import get_settings


def setup_logging() -> None:
    """Configure the root logger. Call once at application startup."""
    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    )

    root = logging.getLogger()
    root.handlers.clear()  # avoid duplicate lines on --reload
    root.addHandler(handler)
    root.setLevel(level)

    # Keep noisy third-party loggers quieter
    logging.getLogger("pymongo").setLevel(logging.WARNING)