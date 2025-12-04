"""
Logger - Centralized logging system for the application.
"""

import logging
import os
from datetime import datetime


class AppLogger:
    """Centralized application logger."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name, log_level=logging.INFO):
        """
        Get or create a logger instance.
        
        Args:
            name (str): Logger name (usually __name__)
            log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = cls._setup_logger(name, log_level)
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _setup_logger(cls, name, log_level):
        """
        Setup logger with file and console handlers.
        
        Args:
            name (str): Logger name
            log_level: Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with date
        log_file = os.path.join(
            log_dir, 
            f"gym_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler (only show WARNING and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def set_level(cls, level):
        """
        Set logging level for all loggers.
        
        Args:
            level: Logging level (INFO, DEBUG, WARNING, ERROR)
        """
        for logger in cls._loggers.values():
            logger.setLevel(level)
