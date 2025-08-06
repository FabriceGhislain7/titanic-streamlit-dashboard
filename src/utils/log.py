# utils/project_logger.py
"""
Logger centralizzato per progetti Python
(versione rinominata da utils/log.py)
"""

import logging
from src.config import LOGGING_CONFIG, DEBUG_MODE

class ProjectLogger:
    """Logger principale per il progetto"""
    
    def __init__(self, name=None):
        self.name = name or self._get_caller_name()
        self.logger = logging.getLogger(self.name)
        self._setup_logger()

    def _get_caller_name(self):
        """Ottiene il nome del chiamante automaticamente"""
        import inspect
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        return module.__name__ if module else __name__

    def _setup_logger(self):
        """Configura gli handler del logger"""
        if not self.logger.handlers:
            self._configure_handlers()

    def _configure_handlers(self):
        """Aggiunge handler solo se non esistono già"""
        config = self._get_dynamic_config()
        
        if config['console']:
            ch = logging.StreamHandler()
            ch.setLevel(config['console_level'])
            ch.setFormatter(logging.Formatter(LOGGING_CONFIG['formatters']['standard']['format']))
            self.logger.addHandler(ch)
        
        if config['file']:
            fh = logging.FileHandler(
                filename=LOGGING_CONFIG['handlers']['file']['filename'],
                encoding='utf-8'
            )
            fh.setLevel(config['file_level'])
            fh.setFormatter(logging.Formatter(LOGGING_CONFIG['formatters']['detailed']['format']))
            self.logger.addHandler(fh)

        self.logger.setLevel(config['global_level'])

    def _get_dynamic_config(self):
        """Restituisce la configurazione in base a DEBUG_MODE"""
        return {
            'console': True,
            'file': True,
            'console_level': 'DEBUG' if DEBUG_MODE else LOGGING_CONFIG['handlers']['console']['level'],
            'file_level': LOGGING_CONFIG['handlers']['file']['level'],
            'global_level': 'DEBUG' if DEBUG_MODE else LOGGING_CONFIG['loggers']['']['level']
        }

    def log(self, level, message, context=None):
        if context:
            message = f"[{context}] {message}"
        self.logger.log(level, message)

    def debug(self, message, context=None):
        self.log(logging.DEBUG, message, context)
    
    def info(self, message, context=None):
        self.log(logging.INFO, message, context)
    
    def warning(self, message, context=None):
        self.log(logging.WARNING, message, context)
    
    def error(self, message, context=None):
        self.log(logging.ERROR, message, context)

def setup_logging():
    """Configura il sistema di logging globale"""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging configurato correttamente")
    return logger

# Logger globale preconfigurato
logger = ProjectLogger()