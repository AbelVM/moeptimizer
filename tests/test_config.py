"""Tests for config module."""

import os

from moeptimizer.config import AppConfig, get_config


class TestConfig:
    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.server.url == "http://localhost:13305/api/v1"
        assert config.server.embed_url == "http://localhost:13305/api/v1"
        assert config.server.llm_model == "Qwen3.6-35B-A3B-MTP-GGUF"
        assert config.agentic.keep_full_steps == 3
        assert config.agentic.max_optimized_chars == 12000
        assert config.agentic.max_optimized_tokens == 3000
        assert config.agentic.proactive_trim_ratio == 0.45
        assert config.agentic.compaction_trigger_ratio == 0.75
        assert config.agentic.fast_path_enabled is True
        assert config.agentic.optimize_code_blocks is False
        assert config.agentic.code_skeleton_enabled is True
        assert config.agentic.semantic_dedup_enabled is False
        assert config.agentic.static_layer_alignment_enabled is False
        assert config.agentic.reasoning_preseed_enabled is False

    def test_env_override(self) -> None:
        os.environ["MOEPT_SERVER__URL"] = "http://test:9999/api/v1"
        os.environ["MOEPT_SERVER__EMBED_URL"] = "http://embed:9999/api/v1"
        os.environ["MOEPT_AGENTIC__KEEP_FULL_STEPS"] = "5"
        try:
            config = AppConfig()
            assert config.server.url == "http://test:9999/api/v1"
            assert config.server.embed_url == "http://embed:9999/api/v1"
            assert config.agentic.keep_full_steps == 5
        finally:
            del os.environ["MOEPT_SERVER__URL"]
            del os.environ["MOEPT_SERVER__EMBED_URL"]
            del os.environ["MOEPT_AGENTIC__KEEP_FULL_STEPS"]

    def test_get_config(self) -> None:
        config = get_config()
        assert isinstance(config, AppConfig)
