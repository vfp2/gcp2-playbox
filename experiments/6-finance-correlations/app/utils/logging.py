from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach any extra contextual fields
        for key, value in record.__dict__.items():
            if key in {"args", "msg", "levelname", "levelno", "pathname", "filename", "module",
                       "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created",
                       "msecs", "relativeCreated", "thread", "threadName", "processName", "process",
                       "name"}:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)




