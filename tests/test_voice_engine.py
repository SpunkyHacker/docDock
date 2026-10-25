"""
Unit tests for Single-Pass Voice Engine with Groq Whisper mocking.
"""

from unittest.mock import MagicMock, patch
from core.voice_engine import process_voice_modality
from core.ltr import rank_doctors


def test_voice_tanglish_mocked_to_dermatology_and_aravind_swaminathan(text_engine, tmp_path):
    """
    Test Case: Spoken Tanglish 'enakku rendu naala skin-la romba itching and red patches irukku'
    Mocked to: 'Doctor, for two days I have severe itching and red patches on my skin.'
    Result: General Dermatology & Cosmetology predicted, Dr. Aravind Swaminathan ranked #1.
    """
    # Create a dummy temporary wav file
    dummy_wav = tmp_path / "tanglish_test.wav"
    dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_translation_response = MagicMock()
    mock_translation_response.text = "Doctor, for two days I have severe itching and red patches on my skin."

    with patch("core.voice_engine.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.audio.translations.create.return_value = mock_translation_response

        voice_out = process_voice_modality(
            audio_filepath=str(dummy_wav),
            groq_api_key="mock_groq_api_key",
            hybrid_text_engine=text_engine
        )

        assert voice_out["transcript"] == "Doctor, for two days I have severe itching and red patches on my skin."
        assert voice_out["error"] is None

        probs = voice_out["probs"]
        top_dept = max(probs.items(), key=lambda x: x[1])[0]
        assert top_dept == "General Dermatology & Cosmetology"

        # Check Doctor Ranking
        ranked_df = rank_doctors(probs, voice_out["transcript"])
        assert not ranked_df.empty
        top_doctor = ranked_df.iloc[0]["Doctor Name"]
        assert "Aravind Swaminathan" in top_doctor


def test_voice_missing_key_graceful_handling(text_engine, tmp_path):
    """
    Test Case: Missing Groq API key returns graceful status without raising unhandled exceptions.
    """
    dummy_wav = tmp_path / "test.wav"
    dummy_wav.write_bytes(b"RIFF dummy audio data")

    voice_out = process_voice_modality(
        audio_filepath=str(dummy_wav),
        groq_api_key="",
        hybrid_text_engine=text_engine
    )

    assert voice_out["transcript"] == ""
    assert voice_out["error"] == "Missing GROQ_API_KEY"
    assert "Groq API key not configured" in voice_out["status"]
