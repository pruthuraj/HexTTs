"""Mel normalization pipeline tests.

Validates that training normalization, inference inverse, and vocoder
re-normalization are internally consistent and cover the full [0, 1] range.
"""

import numpy as np
import pytest


REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100


def dataset_normalize(mel_db: np.ndarray, min_level_db: float) -> np.ndarray:
    """Mirror of raw_dataset.py _audio_to_mel normalization."""
    norm = (mel_db - min_level_db) / -min_level_db
    return np.clip(norm, 0.0, 1.0)


def inference_inverse(norm: np.ndarray, min_level_db: float) -> np.ndarray:
    """Mirror of pipeline.py generate_mel_spectrogram inverse normalization."""
    return norm * -min_level_db + min_level_db


def vocoder_renorm(mel_db: np.ndarray, min_level_db: float) -> np.ndarray:
    """Mirror of pipeline.py mel_spectrogram_to_audio vocoder normalization."""
    return np.clip((mel_db - min_level_db) / -min_level_db, 0.0, 1.0)


class TestDatasetNormalization:
    def test_peak_energy_maps_to_one(self):
        """0 dB (peak energy) must normalize to 1.0."""
        assert dataset_normalize(np.array([0.0]), MIN_LEVEL_DB)[0] == pytest.approx(1.0)

    def test_floor_energy_maps_to_zero(self):
        """min_level_db must normalize to 0.0."""
        assert dataset_normalize(np.array([MIN_LEVEL_DB]), MIN_LEVEL_DB)[0] == pytest.approx(0.0)

    def test_midpoint_is_half(self):
        """Midpoint between min_level_db and 0 dB must normalize to 0.5."""
        mid = MIN_LEVEL_DB / 2.0  # -50 dB
        assert dataset_normalize(np.array([mid]), MIN_LEVEL_DB)[0] == pytest.approx(0.5)

    def test_output_range_is_zero_to_one(self):
        """Realistic speech mel_db range [-80, 0] must stay in [0, 1] after normalization."""
        mel_db = np.linspace(-80.0, 0.0, 100)
        norm = dataset_normalize(mel_db, MIN_LEVEL_DB)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_output_is_not_constant(self):
        """Normalization must not collapse to a constant value (regression guard)."""
        mel_db = np.linspace(-80.0, 0.0, 100)
        norm = dataset_normalize(mel_db, MIN_LEVEL_DB)
        assert norm.std() > 0.01, "Mel normalization collapsed to a near-constant — formula bug"

    def test_old_broken_formula_fails_constant_check(self):
        """The old formula (ref_level_db=20) must be detected as broken."""
        mel_db = np.linspace(-80.0, 0.0, 100)
        broken = np.clip((mel_db - REF_LEVEL_DB) / -REF_LEVEL_DB, 0.0, 1.0)
        assert broken.std() < 1e-9, "Old formula should collapse to constant 1.0"


class TestInverseNormalization:
    def test_round_trip_identity(self):
        """normalize → inverse must recover the original mel_db values."""
        mel_db = np.array([-80.0, -50.0, -20.0, -10.0, 0.0])
        norm = dataset_normalize(mel_db, MIN_LEVEL_DB)
        recovered = inference_inverse(norm, MIN_LEVEL_DB)
        np.testing.assert_allclose(recovered, mel_db, atol=1e-6)

    def test_norm_one_gives_zero_db(self):
        assert inference_inverse(np.array([1.0]), MIN_LEVEL_DB)[0] == pytest.approx(0.0)

    def test_norm_zero_gives_min_level_db(self):
        assert inference_inverse(np.array([0.0]), MIN_LEVEL_DB)[0] == pytest.approx(MIN_LEVEL_DB)


class TestVocoderRenormalization:
    def test_zero_db_maps_to_one(self):
        """Peak energy (0 dB) must give 1.0 at vocoder input."""
        assert vocoder_renorm(np.array([0.0]), MIN_LEVEL_DB)[0] == pytest.approx(1.0)

    def test_min_level_db_maps_to_zero(self):
        """Floor energy must give 0.0 at vocoder input."""
        assert vocoder_renorm(np.array([MIN_LEVEL_DB]), MIN_LEVEL_DB)[0] == pytest.approx(0.0)

    def test_full_pipeline_is_identity(self):
        """dataset_normalize → inverse → vocoder_renorm must recover normalized values."""
        norm_original = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mel_db = inference_inverse(norm_original, MIN_LEVEL_DB)
        norm_recovered = vocoder_renorm(mel_db, MIN_LEVEL_DB)
        np.testing.assert_allclose(norm_recovered, norm_original, atol=1e-6)
