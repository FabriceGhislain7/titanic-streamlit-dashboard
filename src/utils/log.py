"""
Logger centralizzato per progetti Python
(versione rinominata da utils/log.py)
"""

import logging
import inspect

class ProjectLogger:
    """Logger principale per il progetto"""
    
    def __init__(self, name=None):
        self.name = name or self._get_caller_name()
        self.logger = logging.getLogger(self.name)
        self._setup_logger()
        # Disabilita il logging ricorsivo
        self._logging_disabled = False

    def _get_caller_name(self):
        """Ottiene il nome del chiamante automaticamente"""
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
            ch.setFormatter(logging.Formatter(config['formatters']['standard']['format']))
            self.logger.addHandler(ch)
        
        if config['file']:
            fh = logging.FileHandler(
                filename=config['handlers']['file']['filename'],
                encoding='utf-8'
            )
            fh.setLevel(config['file_level'])
            fh.setFormatter(logging.Formatter(config['formatters']['detailed']['format']))
            self.logger.addHandler(fh)

        self.logger.setLevel(config['global_level'])

    def _get_dynamic_config(self):
        """Restituisce la configurazione in base a DEBUG_MODE"""
        from src.config import DEBUG_MODE, LOGGING_CONFIG
        return {
            'console': True,
            'file': True,
            'console_level': 'DEBUG' if DEBUG_MODE else LOGGING_CONFIG['handlers']['console']['level'],
            'file_level': LOGGING_CONFIG['handlers']['file']['level'],
            'global_level': 'DEBUG' if DEBUG_MODE else LOGGING_CONFIG['loggers']['']['level'],
            'formatters': LOGGING_CONFIG['formatters'],
            'handlers': LOGGING_CONFIG['handlers']
        }

    def log(self, level, message, context=None):
        if self._logging_disabled:
            return
            
        self._logging_disabled = True
        try:
            if context:
                message = f"[{context}] {message}"
            self.logger.log(level, message)
        finally:
            self._logging_disabled = False

    def debug(self, message, context=None):
        self.log(logging.DEBUG, message, context)
    
    def info(self, message, context=None):
        self.log(logging.INFO, message, context)
    
    def warning(self, message, context=None):
        self.log(logging.WARNING, message, context)
    
    def error(self, message, context=None):
        self.log(logging.ERROR, message, context)

# Configurazione iniziale senza causare ricorsione
try:
    logger = ProjectLogger(__name__)
    logger._logging_disabled = True
    logger.info(f"Logger globale configurato in {__name__}")
    logger._logging_disabled = False
except Exception as e:
    print(f"Errore durante l'inizializzazione del logger: {str(e)}")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())