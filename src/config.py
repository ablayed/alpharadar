from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: SecretStr
    telegram_chat_id: str

    finnhub_api_key: SecretStr
    coingecko_api_key: SecretStr
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    briefing_hour: int = 7
    briefing_minute: int = 0
    timezone: str = "Europe/London"

    db_path: str = "data/alpharadar.db"

    finnhub_rpm: int = 45
    coingecko_rpm: int = 25
    defillama_rpm: int = 400
    edgar_rps: int = 8

    coingecko_monthly_limit: int = 10_000
    coingecko_warning_threshold: float = 0.8

    @staticmethod
    def _configured_secret(secret: SecretStr | None) -> SecretStr | None:
        if secret is None:
            return None
        return secret if secret.get_secret_value().strip() else None

    @property
    def has_groq_api_key(self) -> bool:
        return self._configured_secret(self.groq_api_key) is not None

    @property
    def has_openrouter_api_key(self) -> bool:
        return self._configured_secret(self.openrouter_api_key) is not None

    @property
    def llm_api_key(self) -> SecretStr | None:
        """Return the first configured multi-provider key."""
        return self._configured_secret(self.openrouter_api_key) or self._configured_secret(self.groq_api_key)


settings = Settings()  # type: ignore[call-arg]
