"""Tests for config module."""

import os

from moeptimizer.config import (
    AppConfig,
    apply_quality_profile,
    get_config,
)


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
        assert config.agentic.quality_profile == "balanced"
        assert config.agentic.explain_mode_enabled is False

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


class TestQualityProfile:
    def test_balanced_is_default(self) -> None:
        config = AppConfig()
        apply_quality_profile(config)
        assert config.agentic.keep_full_steps == 3
        assert config.agentic.max_optimized_tokens == 3000
        assert config.agentic.code_skeleton_enabled is True

    def test_quality_profile_maximizes_fidelity(self) -> None:
        config = AppConfig()
        config.agentic.quality_profile = "quality"
        apply_quality_profile(config)
        assert config.v050.hierarchical_summary_enabled is False
        assert config.agentic.rag_enabled is False
        assert config.agentic.code_skeleton_enabled is False
        assert config.agentic.reasoning_preseed_enabled is False
        assert config.agentic.keep_full_steps == 6
        assert config.agentic.max_optimized_tokens == 6000

    def test_aggressive_profile_saves_more(self) -> None:
        config = AppConfig()
        config.agentic.quality_profile = "aggressive"
        apply_quality_profile(config)
        assert config.agentic.keep_full_steps == 2
        assert config.agentic.max_optimized_tokens == 2000
        assert config.agentic.proactive_trim_ratio == 0.35

    def test_unknown_profile_falls_back_to_balanced(self) -> None:
        config = AppConfig()
        config.agentic.quality_profile = "bogus"
        apply_quality_profile(config)
        assert config.agentic.quality_profile == "balanced"
        assert config.agentic.max_optimized_tokens == 3000


class TestConfigCheck:
    def test_clean_config_has_no_errors(self) -> None:
        from moeptimizer.config_check import check_config

        config = AppConfig()
        apply_quality_profile(config)
        issues = check_config(config)
        assert not any(i.severity == "ERROR" for i in issues)

    def test_bad_trim_order_is_error(self) -> None:
        from moeptimizer.config_check import check_config

        config = AppConfig()
        config.agentic.proactive_trim_ratio = 0.9
        config.agentic.compaction_trigger_ratio = 0.5
        issues = check_config(config)
        codes = {i.code for i in issues}
        assert "trim_order" in codes

    def test_summary_breaks_prefix_warns(self) -> None:
        from moeptimizer.config_check import check_config

        config = AppConfig()
        config.v050.hierarchical_summary_enabled = True
        issues = check_config(config)
        codes = {i.code for i in issues}
        assert "summary_breaks_prefix" in codes

    def test_low_water_out_of_range_is_error(self) -> None:
        from moeptimizer.config_check import check_config

        config = AppConfig()
        config.agentic.eviction_low_water_ratio = 1.5
        issues = check_config(config)
        codes = {i.code for i in issues}
        assert "low_water" in codes
