from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Thresholds(BaseModel):
    """Threshold configuration for mapping anomaly scores to directions.

    Per Scott Wilber (canon.yaml), larger sustained deviations from the expected
    random baseline (Max[Z]) indicate stronger collective effects; we translate
    these to directional predictions with confidence bands.
    """

    up_threshold: float = Field(1.4, description="Max[Z] >= this maps to 'UP'")
    down_threshold: float = Field(1.4, description="Max[Z] >= this maps to 'DOWN'")
    min_confidence: float = Field(
        0.5, description="Minimum confidence to accept a prediction (0..1)"
    )


class MethodConfig(BaseModel):
    method: str = Field(
        "maxz",
        description="Selected anomaly method key (extensible via app.core.methods)",
    )
    expected_mean: float = 100.0
    expected_std: float = 7.0712
    window_size: int = Field(300, description="Number of recent sensor samples to use")


class RuntimeConfig(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["SPY", "IVV", "VOO", "VXX", "UVXY"])
    sensor_buffer_size: int = 10_000
    market_buffer_size: int = 10_000
    bin_duration_sec: int = Field(60, description="Cadence for predictions")
    horizon_sec: int = Field(300, description="Prediction horizon in seconds")
    thresholds: Thresholds = Field(default_factory=Thresholds)
    method: MethodConfig = Field(default_factory=MethodConfig)
    network_timeout_sec: int = 10
    max_retries: int = 5
    backoff_base_sec: float = 0.5
    backoff_cap_sec: float = 10.0
    dev_mode: bool = True


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Networking / Service
    DASH_HOST: str = "0.0.0.0"
    DASH_PORT: int = 8050
    LOG_LEVEL: str = "INFO"
    SIMULATE: bool = True

    # Alpaca
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"


class AppConfig(BaseModel):
    env: EnvSettings
    runtime: RuntimeConfig

    @staticmethod
    def load(config_path: Optional[Path] = None) -> "AppConfig":
        env = EnvSettings()  # loads from environment and .env

        runtime = RuntimeConfig()
        if config_path is None:
            default_path = Path("config.yaml")
            config_path = default_path if default_path.exists() else None

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            try:
                runtime = RuntimeConfig(**raw)
            except ValidationError as ve:
                raise ValueError(f"Invalid config.yaml: {ve}")

        # If no Alpaca keys and SIMULATE not explicitly set, default to simulate
        simulate = env.SIMULATE or not (
            env.ALPACA_API_KEY and env.ALPACA_SECRET_KEY
        )
        # Propagate to runtime.dev_mode
        runtime.dev_mode = simulate
        return AppConfig(env=env, runtime=runtime)


def load_config() -> AppConfig:
    """Load merged configuration from environment and optional YAML.

    Per Scott Wilber (canon.yaml), defaults aim for conservative runtime behavior
    with simulation fallback to ensure deterministic reproduction during research.
    """

    return AppConfig.load()




