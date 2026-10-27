"""Tests for spiral_03_orchestration.tension_profile — profiles and configuration."""

import pytest

from spiral_03_orchestration.tension_profile import (
    PROFILES,
    ProfileType,
    get_profile,
)


class TestProfileType:
    """Tests for ProfileType enum."""

    def test_all_profiles_exist(self):
        """All 4 profile types should be defined."""
        assert len(ProfileType) == 4
        assert ProfileType.EXPLORATORY.value == "EXPLORATORY"
        assert ProfileType.STANDARD.value == "STANDARD"
        assert ProfileType.CRITICAL.value == "CRITICAL"
        assert ProfileType.DESTRUCTIVE.value == "DESTRUCTIVE"

    def test_profiles_are_strings(self):
        """ProfileType values should be strings."""
        for p in ProfileType:
            assert isinstance(p.value, str)


class TestTensionProfile:
    """Tests for TensionProfile dataclass."""

    def test_profile_is_frozen(self):
        """Profiles should be immutable."""
        profile = get_profile("STANDARD")
        with pytest.raises(AttributeError):
            profile.creativity_weight = 0.9  # type: ignore

    def test_to_dict(self):
        """to_dict should include all config keys."""
        profile = get_profile("STANDARD")
        d = profile.to_dict()
        assert "name" in d
        assert "creativity_weight" in d
        assert "logic_weight" in d
        assert "confidence_threshold" in d
        assert "max_chunks" in d
        assert "use_reranker" in d


class TestProfiles:
    """Tests for the 4 pre-built profiles."""

    def test_exploratory_weights(self):
        """Exploratory: high creativity, low logic."""
        p = PROFILES[ProfileType.EXPLORATORY]
        assert p.creativity_weight > p.logic_weight
        assert p.creativity_weight == pytest.approx(0.8)
        assert p.logic_weight == pytest.approx(0.2)

    def test_exploratory_threshold_is_lowest(self):
        """Exploratory should have the lowest confidence threshold."""
        p = PROFILES[ProfileType.EXPLORATORY]
        assert p.confidence_threshold == pytest.approx(0.15)

    def test_exploratory_no_reranker(self):
        """Exploratory does not use reranker (freedom)."""
        p = PROFILES[ProfileType.EXPLORATORY]
        assert p.use_reranker is False

    def test_standard_balanced(self):
        """Standard: equal creativity and logic weights."""
        p = PROFILES[ProfileType.STANDARD]
        assert p.creativity_weight == pytest.approx(p.logic_weight)
        assert p.creativity_weight == pytest.approx(0.5)

    def test_standard_uses_reranker(self):
        """Standard uses reranker."""
        p = PROFILES[ProfileType.STANDARD]
        assert p.use_reranker is True

    def test_critical_weights(self):
        """Critical: low creativity, high logic."""
        p = PROFILES[ProfileType.CRITICAL]
        assert p.creativity_weight < p.logic_weight
        assert p.logic_weight == pytest.approx(0.8)

    def test_critical_high_threshold(self):
        """Critical has a high confidence threshold."""
        p = PROFILES[ProfileType.CRITICAL]
        assert p.confidence_threshold == pytest.approx(0.6)

    def test_critical_more_chunks(self):
        """Critical consults more chunks for rigor."""
        p = PROFILES[ProfileType.CRITICAL]
        assert p.max_chunks == 8

    def test_destructive_most_restrictive(self):
        """Destructive should have the highest confidence threshold."""
        p = PROFILES[ProfileType.DESTRUCTIVE]
        assert p.confidence_threshold == pytest.approx(0.8)

    def test_destructive_weights(self):
        """Destructive: minimal creativity, maximum logic."""
        p = PROFILES[ProfileType.DESTRUCTIVE]
        assert p.creativity_weight == pytest.approx(0.1)
        assert p.logic_weight == pytest.approx(0.9)

    def test_destructive_fewest_chunks(self):
        """Destructive uses fewest chunks (maximum caution)."""
        p = PROFILES[ProfileType.DESTRUCTIVE]
        assert p.max_chunks == 3

    def test_threshold_ordering(self):
        """Thresholds should increase: EXPLORATORY < STANDARD < CRITICAL < DESTRUCTIVE."""
        thresholds = [PROFILES[p].confidence_threshold for p in ProfileType]
        assert thresholds == sorted(thresholds)

    def test_all_profiles_have_descriptions(self):
        """Every profile must have a non-empty description."""
        for p in PROFILES.values():
            assert p.description
            assert len(p.description) > 10


class TestGetProfile:
    """Tests for the get_profile() function."""

    def test_get_by_string(self):
        """Should accept case-insensitive string."""
        p = get_profile("standard")
        assert p.name == ProfileType.STANDARD

    def test_get_by_uppercase_string(self):
        """Should accept uppercase string."""
        p = get_profile("CRITICAL")
        assert p.name == ProfileType.CRITICAL

    def test_get_by_enum(self):
        """Should accept ProfileType enum."""
        p = get_profile(ProfileType.EXPLORATORY)
        assert p.name == ProfileType.EXPLORATORY

    def test_get_by_enum_all_profiles(self):
        """Every ProfileType enum value should resolve correctly."""
        for profile_type in ProfileType:
            p = get_profile(profile_type)
            assert p.name == profile_type

    def test_get_invalid_raises(self):
        """Invalid name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("NONEXISTENT")
