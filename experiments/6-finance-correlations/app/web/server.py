from __future__ import annotations

from typing import Optional

from ..config import AppConfig, load_config
from .dashboard import build_dash_app


def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    cfg = load_config()
    app = build_dash_app(cfg)
    app.run_server(
        host or cfg.env.DASH_HOST,
        cfg.env.DASH_PORT if port is None else port,
        debug=False,
    )

if __name__ == "__main__":  # pragma: no cover
    serve()




