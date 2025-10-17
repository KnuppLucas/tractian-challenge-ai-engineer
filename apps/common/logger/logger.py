import sys
import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """
    Formata logs como JSON estruturado.
    Ideal para uso em containers e observabilidade (Loki, ELK, Datadog).
    """

    def format(self, record):
        """
        Configura um logger global padronizado (JSON + stdout).
        Evita duplicação de handlers e permite reuso em todos os serviços.
        """
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "service"):
            log_record["service"] = record.service

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(level=logging.INFO):
    """
    Configura um logger global padronizado (JSON + stdout).
    Evita duplicação de handlers e permite reuso em todos os serviços.
    """
    logger = logging.getLogger("rag")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logging(logging.DEBUG)
