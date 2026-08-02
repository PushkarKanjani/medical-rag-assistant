from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    def Field(default: object = None, **_: object) -> object:
        return default


    class BaseSettings:
        model_config: dict[str, object] = {}

        def __init__(self, **overrides: object) -> None:
            values = self._load_values()
            values.update(overrides)
            for field_name in self.__annotations__:
                setattr(self, field_name, values.get(field_name, getattr(self.__class__, field_name, None)))

        @classmethod
        def _load_values(cls) -> dict[str, str]:
            values: dict[str, str] = {}
            config = getattr(cls, "model_config", {})
            env_file = config.get("env_file") if isinstance(config, dict) else None
            if env_file:
                env_path = Path(env_file)
                if env_path.exists():
                    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, raw_value = line.split("=", 1)
                        values[key.strip()] = raw_value.strip().strip('"').strip("'")
            values.update(os.environ)
            return values


    def SettingsConfigDict(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    supabase_url: str = Field(default="")
    supabase_service_key: str = Field(default="")
    postgres_dsn: str = Field(default="")
    groq_api_key: str = Field(default="")
    deepseek_api_key: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    abdm_sandbox_base_url: str = Field(default="")
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()