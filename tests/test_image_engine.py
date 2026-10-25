"""
Unit tests for Hybrid Medical Image Analyzer with BiomedCLIP local fallback and Groq Vision.
"""

from unittest.mock import MagicMock, patch
from core.image_engine import HybridMedicalImageAnalyzer


def test_image_missing_groq_key_biomedclip_fallback(image_engine, sample_pil_image, text_engine):
    """
    Test Case: When Groq API key is empty/missing, image analyzer falls back to 100% BiomedCLIP.
    Does not crash, returns valid probability distribution summing to 1.0.
    """
    out = image_engine.analyze_image(
        pil_image=sample_pil_image,
        hybrid_text_engine=text_engine,
        groq_api_key=""
    )

    assert out["error"] is None
    assert "BiomedCLIP" in out["status"]
    assert len(out["probs"]) == 9
    prob_sum = sum(out["probs"].values())
    assert abs(prob_sum - 1.0) < 1e-3


def test_image_groq_vision_mocked_fusion(image_engine, sample_pil_image, text_engine):
    """
    Test Case: When Groq Vision succeeds, fuses 50% BiomedCLIP + 50% Visual-Text.
    """
    mock_choice = MagicMock()
    mock_choice.message.content = "Clinical photograph of erythematous scaly skin rash with red plaques."
    mock_vlm_response = MagicMock()
    mock_vlm_response.choices = [mock_choice]

    with patch("core.image_engine.Groq") as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_vlm_response

        out = image_engine.analyze_image(
            pil_image=sample_pil_image,
            hybrid_text_engine=text_engine,
            groq_api_key="mock_groq_api_key"
        )

        assert out["visual_description"] == "Clinical photograph of erythematous scaly skin rash with red plaques."
        assert "BiomedCLIP (50%)" in out["status"]
        assert len(out["probs"]) == 9
        assert abs(sum(out["probs"].values()) - 1.0) < 1e-3
