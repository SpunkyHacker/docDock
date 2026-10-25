"""
Pytest configuration and session-scoped fixtures for docDock.
Pre-loads models once across the test suite for fast, reliable testing.
"""

import pytest
from PIL import Image
from core.text_engine import HybridMedicalTextClassifier
from core.image_engine import HybridMedicalImageAnalyzer


@pytest.fixture(scope="session")
def text_engine():
    """Session-scoped hybrid text classifier instance."""
    return HybridMedicalTextClassifier()


@pytest.fixture(scope="session")
def image_engine():
    """Session-scoped hybrid image analyzer instance."""
    return HybridMedicalImageAnalyzer()


@pytest.fixture
def sample_pil_image():
    """Returns a test PIL image."""
    return Image.new("RGB", (224, 224), color=(200, 70, 70))
