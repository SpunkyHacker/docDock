"""
End-to-End integration tests for docDock clinical pipeline.
"""

from unittest.mock import MagicMock, patch
from PIL import Image
from app import run_docdock_pipeline


def test_pipeline_end_to_end_multimodal():
    """
    Runs the full docDock pipeline with text, voice (mocked), and image (mocked),
    verifying output structure and consistency.
    """
    mock_trans_resp = MagicMock()
    mock_trans_resp.text = "I have had severe redness in my eye and blurry vision since yesterday."

    mock_vlm_choice = MagicMock()
    mock_vlm_choice.message.content = "Clinical photograph of acute conjunctival injection and red eye."
    mock_vlm_resp = MagicMock()
    mock_vlm_resp.choices = [mock_vlm_choice]

    test_img = Image.new("RGB", (224, 224), color=(255, 0, 0))

    with patch("core.voice_engine.Groq") as mock_groq_voice, \
         patch("core.image_engine.Groq") as mock_groq_vision:

        mock_v_client = MagicMock()
        mock_groq_voice.return_value = mock_v_client
        mock_v_client.audio.translations.create.return_value = mock_trans_resp

        mock_vis_client = MagicMock()
        mock_groq_vision.return_value = mock_vis_client
        mock_vis_client.chat.completions.create.return_value = mock_vlm_resp

        (
            banner_html,
            top_card,
            top3_dict,
            weights_display,
            transcript,
            vis_desc,
            doctor_df,
            audit
        ) = run_docdock_pipeline(
            typed_text="my eye is swollen and watery",
            audio_filepath="sample_data/tanglish_dermatology_symptom.wav",
            pil_image=test_img,
            w1_text=1.0,
            w2_speech=1.0,
            w3_image=1.0,
            ui_groq_key="mock_key"
        )

        assert "Safety Triage Passed" in banner_html
        assert "Recommended Hospital Department" in top_card
        assert len(top3_dict) <= 3
        assert "Text (ŵ₁)" in weights_display
        assert "blurry vision" in transcript
        assert "red eye" in vis_desc
        assert not doctor_df.empty
        assert "Doctor Name" in doctor_df.columns
        assert "LTR Score" in doctor_df.columns


def test_pipeline_end_to_end_general_medicine_fallback():
    """
    Runs the pipeline on vague text 'I don't feel well', verifying that the General Medicine
    fallback triggers with fallback_reason='vague_input' and General Medicine doctors are ranked #1.
    """
    (
        banner_html,
        top_card,
        top3_dict,
        weights_display,
        transcript,
        vis_desc,
        doctor_df,
        audit
    ) = run_docdock_pipeline(
        typed_text="I don't feel well",
        audio_filepath=None,
        pil_image=None,
        w1_text=1.0,
        w2_speech=0.0,
        w3_image=0.0,
        ui_groq_key=""
    )

    assert "Safety Triage Passed" in banner_html
    assert "General Medicine (Internal Medicine)" in top_card
    assert "vague_input" in top_card or "vague_input" in audit
    assert not doctor_df.empty
    assert doctor_df.iloc[0]["Department"] == "General Medicine (Internal Medicine)"
