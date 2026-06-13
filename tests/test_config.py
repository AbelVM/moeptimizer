"""Tests for config module."""

import os

from moeptimizer.config import AppConfig, get_config


class TestConfig:
    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.server.url == "http://localhost:13305/api/v1"
        assert config.server.llm_model == "Qwen3.6-35B-A3B-MTP-GGUF"
        assert config.agentic.keep_full_steps == 3
        assert config.agentic.max_optimized_chars == 12000

    def test_env_override(self) -> None:
        os.environ["MOEPT_SERVER__URL"] = "http://test:9999/api/v1"
        os.environ["MOEPT_AGENTIC__KEEP_FULL_STEPS"] = "5"
        try:
            config = AppConfig()
            assert config.server.url == "http://test:9999/api/v1"
            assert config.agentic.keep_full_steps == 5
        finally:
            del os.environ["MOEPT_SERVER__URL"]
            del os.environ["MOEPT_AGENTIC__KEEP_FULL_STEPS"]

    def test_get_config(self) -> None:
        config = get_config()
        assert isinstance(config, AppConfig)
