"""
Single-Pass Voice Engine for Multilingual Speech Symptom Ingestion.
Uses Groq Whisper-Large-v3 via audio.translations API to translate Tamil, Hindi,
Tanglish, Hinglish, and English directly into clinical English text in a single ~0.3s pass,
then routes the transcript into the Local Hybrid Medical Text Engine.
"""

import os
from typing import Dict, Any, Optional
from groq import Groq

from core.config import GROQ_WHISPER_MODEL
from core.text_engine import HybridMedicalTextClassifier


def process_voice_modality(
    audio_filepath: Optional[str],
    groq_api_key: str,
    hybrid_text_engine: HybridMedicalTextClassifier
) -> Dict[str, Any]:
    """
    Translates spoken patient symptoms (Tamil, Hindi, Tanglish, Hinglish, English)
    into English symptom text via Groq whisper-large-v3, then calculates department
    probabilities using the local HybridMedicalTextClassifier.

    Gracefully handles missing keys, network errors, and corrupt audio files.
    """
    specialties = hybrid_text_engine.specialties
    zero_probs = {spec: 0.0 for spec in specialties}

    # Guard: Audio file provided and exists
    if not audio_filepath or not os.path.exists(audio_filepath):
        return {
            "transcript": "",
            "probs": zero_probs,
            "status": "Inactive (No audio input provided)",
            "error": None
        }

    # Guard: Check file size (ensure not empty)
    try:
        if os.path.getsize(audio_filepath) == 0:
            return {
                "transcript": "",
                "probs": zero_probs,
                "status": "Audio file is empty",
                "error": "Empty audio recording"
            }
    except Exception as e:
        return {
            "transcript": "",
            "probs": zero_probs,
            "status": f"Audio file access error: {e}",
            "error": str(e)
        }

    # Guard: Groq API Key
    clean_key = (groq_api_key or "").strip()
    if not clean_key:
        return {
            "transcript": "",
            "probs": zero_probs,
            "status": "Groq API key not configured. Voice translation unavailable.",
            "error": "Missing GROQ_API_KEY"
        }

    # Single-pass translation via Groq whisper-large-v3
    try:
        client = Groq(api_key=clean_key, timeout=30.0)
        with open(audio_filepath, "rb") as audio_file:
            whisper_response = client.audio.translations.create(
                file=(os.path.basename(audio_filepath), audio_file.read()),
                model=GROQ_WHISPER_MODEL,
                prompt=(
                    "Patient describing clinical medical symptoms such as skin rash, "
                    "chest pain, joint swelling, blurry vision, fever, earache, or stomach pain."
                ),
                response_format="json",
                temperature=0.0
            )

        english_transcript = whisper_response.text.strip()
        if not english_transcript:
            return {
                "transcript": "",
                "probs": zero_probs,
                "status": "Whisper audio translation returned empty transcript",
                "error": "Empty transcript"
            }

        # Route English transcript to Local Hybrid Medical Text Engine
        speech_probs = hybrid_text_engine.predict_specialty(english_transcript)

        return {
            "transcript": english_transcript,
            "probs": speech_probs,
            "status": "Groq API (whisper-large-v3 Single-Pass Translation) -> Local Hybrid Text Engine",
            "error": None
        }

    except Exception as e:
        # Graceful degradation without crashing the application
        error_msg = f"Voice transcription failed: {type(e).__name__} - {str(e)}"
        return {
            "transcript": "",
            "probs": zero_probs,
            "status": f"Voice translation error ({type(e).__name__})",
            "error": error_msg
        }
