from __future__ import annotations

from trading_app.config import load_environment


def test_load_environment_defaults_to_dry_run(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("APP_ENV=test\n", encoding="utf-8")

    settings = load_environment(env_file)

    assert settings.dry_run is True
    assert settings.auto_trade is False
    assert settings.alpaca_paper is True
    assert settings.alpaca_api_key == ""
    assert settings.alpaca_secret_key == ""
    assert settings.alpha_vantage_api_key == ""
    assert settings.finnhub_api_key == ""
    assert settings.polygon_api_key == ""
    assert settings.fmp_api_key == ""


def test_load_environment_reads_alpaca_credentials(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ALPACA_API_KEY=paper-key",
                "ALPACA_SECRET_KEY=paper-secret",
                "ALPACA_PAPER=true",
                "DRY_RUN=false",
                "AUTO_TRADE=true",
                "ALPHA_VANTAGE_API_KEY=alpha-key",
                "FINNHUB_API_KEY=finnhub-key",
                "POLYGON_API_KEY=polygon-key",
                "FMP_API_KEY=fmp-key",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_environment(env_file)

    assert settings.alpaca_api_key == "paper-key"
    assert settings.alpaca_secret_key == "paper-secret"
    assert settings.alpaca_paper is True
    assert settings.dry_run is False
    assert settings.auto_trade is True
    assert settings.alpha_vantage_api_key == "alpha-key"
    assert settings.finnhub_api_key == "finnhub-key"
    assert settings.polygon_api_key == "polygon-key"
    assert settings.fmp_api_key == "fmp-key"
