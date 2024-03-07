import logging
import coloredlogs


class AppLogger:
    DEFAULT_LOGGING_LEVEL = logging.DEBUG
    __logging_label: int = None
    __logging_file_name: str = None
    __logger: logging.Logger = None

    @classmethod
    def config(cls, logging_label: int, file_name: str = None) -> None:
        cls.__logging_label = logging_label
        cls.__logging_file_name = file_name
        cls.__logger = logging.getLogger(__name__)
        coloredlogs.install(level=cls.__logging_label, logger=cls.__logger, file_name=cls.__logging_file_name)

    @classmethod
    def getLogger(cls) -> logging.Logger:
        return cls.__logger

