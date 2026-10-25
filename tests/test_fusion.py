"""
Unit tests for Dynamic Weighted Multimodal Late Fusion and General Medicine Fallback Engine.
"""

from core.fusion import (
    dynamic_multimodal_fusion,
    is_vague_or_gibberish,
    apply_general_medicine_fallback,
)
from core.config import GENERAL_MEDICINE_DEPT


def test_fusion_voice_only_normalization():
    """
    Test Case: With only voice supplied, normalized weights must be exactly (0.0, 1.0, 0.0)
    """
    specialties = ["General Dermatology & Cosmetology", "Cardiology & Cardiothoracic"]
    dummy_probs = {s: 0.5 for s in specialties}

    fusion = dynamic_multimodal_fusion(
        text_probs=dummy_probs,
        speech_probs=dummy_probs,
        image_probs=dummy_probs,
        has_text=False,
        has_speech=True,
        has_image=False,
        w1_text=1.0,
        w2_speech=1.0,
        w3_image=1.0,
        specialties=specialties,
        unified_symptom_text="symptom description text"
    )

    assert fusion["normalized_weights"] == (0.0, 1.0, 0.0)
    assert fusion["active_modalities"] == ["Speech"]


def test_fusion_all_zero_sliders_fallback():
    """
    Test Case: With all zero sliders and one active modality, falls back to equal weights
    """
    specialties = ["General Dermatology & Cosmetology", "Cardiology & Cardiothoracic"]
    dummy_probs = {s: 0.5 for s in specialties}

    fusion = dynamic_multimodal_fusion(
        text_probs=dummy_probs,
        speech_probs=dummy_probs,
        image_probs=dummy_probs,
        has_text=True,
        has_speech=False,
        has_image=False,
        w1_text=0.0,
        w2_speech=0.0,
        w3_image=0.0,
        specialties=specialties,
        unified_symptom_text="symptom description text"
    )

    assert fusion["normalized_weights"] == (1.0, 0.0, 0.0)
    assert fusion["active_modalities"] == ["Text"]


def test_fallback_vague_input():
    """
    Test Case: 'I don't feel well' -> General Medicine, fallback_reason = vague_input
    """
    assert is_vague_or_gibberish("I don't feel well") is True

    probs = {
        "General Dermatology & Cosmetology": 0.40,
        "Cardiology & Cardiothoracic": 0.30,
        GENERAL_MEDICINE_DEPT: 0.30
    }
    fusion = dynamic_multimodal_fusion(
        text_probs=probs,
        speech_probs=probs,
        image_probs=probs,
        has_text=True,
        has_speech=False,
        has_image=False,
        unified_symptom_text="I don't feel well"
    )

    assert fusion["fallback"]["is_fallback"] is True
    assert fusion["fallback"]["fallback_reason"] == "vague_input"
    assert fusion["fallback"]["routed_specialty"] == GENERAL_MEDICINE_DEPT
    assert len(fusion["fallback"]["possible_referrals"]) == 2


def test_fallback_gibberish_and_empty():
    """
    Test Case: 'asdf qwerty' or empty input -> General Medicine, fallback_reason = vague_input / no_input
    """
    # Gibberish test
    assert is_vague_or_gibberish("asdf qwerty") is True
    probs = {"Dept A": 0.5, GENERAL_MEDICINE_DEPT: 0.5}
    f_gibberish = dynamic_multimodal_fusion(
        probs, probs, probs, True, False, False, unified_symptom_text="asdf qwerty"
    )
    assert f_gibberish["fallback"]["fallback_reason"] == "vague_input"
    assert f_gibberish["fallback"]["routed_specialty"] == GENERAL_MEDICINE_DEPT

    # Empty input test
    f_empty = dynamic_multimodal_fusion(
        {}, {}, {}, False, False, False, unified_symptom_text=""
    )
    assert f_empty["fallback"]["fallback_reason"] == "no_input"
    assert f_empty["fallback"]["routed_specialty"] == GENERAL_MEDICINE_DEPT


def test_fallback_ambiguous_and_low_confidence():
    """
    Test Case: Mixed query symptoms -> General Medicine via low_confidence or ambiguous margin
    """
    # 1. Ambiguous margin (Top-1 - Top-2 < 0.08)
    probs_ambiguous = {
        "Neurology & Neurosurgery": 0.36,
        "Gastroenterology & Hepatology": 0.34,
        "Orthopaedics & Rheumatology": 0.20,
        GENERAL_MEDICINE_DEPT: 0.10
    }
    f_amb = dynamic_multimodal_fusion(
        probs_ambiguous, probs_ambiguous, probs_ambiguous, True, False, False,
        unified_symptom_text="mild headache and some stomach discomfort and knee pain"
    )
    assert f_amb["fallback"]["is_fallback"] is True
    assert f_amb["fallback"]["fallback_reason"] == "ambiguous"
    assert f_amb["fallback"]["routed_specialty"] == GENERAL_MEDICINE_DEPT

    # 2. Low confidence (< 0.35)
    probs_low_conf = {
        "Neurology & Neurosurgery": 0.26,
        "Gastroenterology & Hepatology": 0.16,
        "Orthopaedics & Rheumatology": 0.15,
        "ENT (Otorhinolaryngology)": 0.15,
        GENERAL_MEDICINE_DEPT: 0.28
    }
    # If GenMed is argmax -> general_is_top
    f_low = dynamic_multimodal_fusion(
        probs_low_conf, probs_low_conf, probs_low_conf, True, False, False,
        unified_symptom_text="mild headache and some stomach discomfort and knee pain"
    )
    assert f_low["fallback"]["routed_specialty"] == GENERAL_MEDICINE_DEPT


def test_fallback_thresholds_tuning():
    """
    Test Case: Modulating LOW_CONFIDENCE_THRESHOLD and AMBIGUITY_MARGIN alters fallback decisions as expected.
    """
    probs = {
        "General Dermatology & Cosmetology": 0.40,
        "ENT (Otorhinolaryngology)": 0.32,
        GENERAL_MEDICINE_DEPT: 0.28
    }
    text = "severe skin itching and red patches and throat irritation"

    # With default threshold 0.35 and margin 0.08 (margin = 0.40 - 0.32 = 0.08 -> not < 0.08)
    # If we increase ambiguity margin to 0.12, it must trigger 'ambiguous' fallback!
    f_tuned = dynamic_multimodal_fusion(
        probs, probs, probs, True, False, False,
        unified_symptom_text=text,
        ambiguity_margin=0.12
    )
    assert f_tuned["fallback"]["is_fallback"] is True
    assert f_tuned["fallback"]["fallback_reason"] == "ambiguous"

    # If we raise low_confidence_threshold to 0.50 (0.40 < 0.50), it must trigger 'low_confidence'
    f_tuned_conf = dynamic_multimodal_fusion(
        probs, probs, probs, True, False, False,
        unified_symptom_text=text,
        low_conf_threshold=0.50,
        ambiguity_margin=0.05
    )
    assert f_tuned_conf["fallback"]["is_fallback"] is True
    assert f_tuned_conf["fallback"]["fallback_reason"] == "low_confidence"


def test_dermatology_no_fallback_when_distinct():
    """
    Test Case: High confidence distinct specialist match -> fallback_reason = none
    """
    probs = {
        "General Dermatology & Cosmetology": 0.65,
        "Cardiology & Cardiothoracic": 0.15,
        GENERAL_MEDICINE_DEPT: 0.10,
        "ENT (Otorhinolaryngology)": 0.10
    }
    text = "itchy red patches on my elbow"
    fusion = dynamic_multimodal_fusion(
        probs, probs, probs, True, False, False,
        unified_symptom_text=text
    )
    assert fusion["fallback"]["is_fallback"] is False
    assert fusion["fallback"]["fallback_reason"] == "none"
    assert fusion["fallback"]["routed_specialty"] == "General Dermatology & Cosmetology"
