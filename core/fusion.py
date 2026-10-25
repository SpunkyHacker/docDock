"""
Dynamic Weighted Multimodal Fusion Engine with General Medicine Fallback Routing.
Fuses active modalities dynamically and evaluates clinical safety fallback rules to route
ambiguous, low-confidence, or non-specific symptoms to General Medicine (Internal Medicine).
"""

import re
from typing import Dict, List, Tuple, Any, Optional

from core.config import (
    GENERAL_MEDICINE_DEPT,
    LOW_CONFIDENCE_THRESHOLD,
    AMBIGUITY_MARGIN,
    MIN_MEANINGFUL_WORDS,
)

# Common conversational stopwords to detect meaningful clinical terms
COMMON_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "feel", "feeling",
    "dont", "don't", "hello", "hi", "hey", "doctor", "please", "help", "need",
    "want", "like", "get", "got", "also", "something", "really"
}


def count_meaningful_words(text: str) -> int:
    """
    Extracts tokens, filters stopwords and short gibberish, and counts meaningful words.
    """
    if not text:
        return 0
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    meaningful = [w for w in words if w not in COMMON_STOPWORDS]
    return len(meaningful)


def is_vague_or_gibberish(text: str, min_words: int = MIN_MEANINGFUL_WORDS) -> bool:
    """
    Returns True if text contains fewer than min_words meaningful clinical terms
    or exhibits repetitive gibberish patterns.
    """
    if not text or not text.strip():
        return True

    clean = text.strip()
    words = re.findall(r"\b[a-zA-Z]{2,}\b", clean.lower())
    if len(words) < 2:
        return True

    # Check for keyboard smash / random gibberish (e.g., 'asdf qwerty', 'zxcvbnm')
    gibberish_patterns = [r"\basdf\b", r"\bqwerty\b", r"\bzxcv\b", r"\bjkl\b"]
    for pat in gibberish_patterns:
        if re.search(pat, clean.lower()):
            return True

    meaningful_count = count_meaningful_words(clean)
    return meaningful_count < min_words


def apply_general_medicine_fallback(
    fused_probabilities: Dict[str, float],
    unified_symptom_text: str,
    active_modalities: List[str],
    low_conf_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN
) -> Dict[str, Any]:
    """
    Evaluates the 5 clinical fallback rules to determine if routing should fall back
    to General Medicine (Internal Medicine):
      1. Top specialty fused probability < LOW_CONFIDENCE_THRESHOLD (default 0.35)
      2. Margin between Top-1 and Top-2 probabilities < AMBIGUITY_MARGIN (default 0.08)
      3. Input too short/vague/meaningless (< 3 meaningful words across text sources) or gibberish
      4. No modality provided or all calls failed
      5. General Medicine is itself the natural argmax

    Returns:
        routed_specialty: str
        is_fallback: bool
        fallback_reason: str ('low_confidence' | 'ambiguous' | 'vague_input' | 'no_input' | 'general_is_top' | 'none')
        fallback_note: str
        possible_referrals: list of tuples [(dept, prob), ...] (top-2 original specialist candidates)
    """
    sorted_specs = sorted(fused_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_specialty, top_confidence = sorted_specs[0] if sorted_specs else (GENERAL_MEDICINE_DEPT, 0.0)
    second_specialty, second_confidence = sorted_specs[1] if len(sorted_specs) > 1 else ("", 0.0)

    # Extract top-2 specialist candidates (excluding General Medicine) as referrals
    possible_referrals = [
        (dept, prob) for dept, prob in sorted_specs
        if dept != GENERAL_MEDICINE_DEPT
    ][:2]

    has_image_evidence = ("Image" in active_modalities)

    # Condition 4: No modality provided or produced usable output
    if not active_modalities:
        return {
            "routed_specialty": GENERAL_MEDICINE_DEPT,
            "routed_confidence": top_confidence,
            "is_fallback": True,
            "fallback_reason": "no_input",
            "fallback_note": (
                "Please describe your symptoms by text, voice, or photo. "
                "We recommend starting with General Medicine for a comprehensive check-up."
            ),
            "possible_referrals": possible_referrals
        }

    # Condition 3: Input is too short, vague, or gibberish (ONLY when there is no image evidence)
    if not has_image_evidence and is_vague_or_gibberish(unified_symptom_text):
        return {
            "routed_specialty": GENERAL_MEDICINE_DEPT,
            "routed_confidence": top_confidence,
            "is_fallback": True,
            "fallback_reason": "vague_input",
            "fallback_note": (
                "Your symptoms were not specific enough to pick a specialist, so we recommend "
                "starting with General Medicine, who can refer you onward."
            ),
            "possible_referrals": possible_referrals
        }

    # Condition 5: General Medicine is naturally the argmax
    if top_specialty == GENERAL_MEDICINE_DEPT:
        return {
            "routed_specialty": GENERAL_MEDICINE_DEPT,
            "routed_confidence": top_confidence,
            "is_fallback": False,
            "fallback_reason": "general_is_top",
            "fallback_note": (
                "Your symptoms indicate a systemic or primary general health condition. "
                "General Medicine (Internal Medicine) is the primary recommended department."
            ),
            "possible_referrals": possible_referrals
        }

    # Condition 1: Low Confidence (< threshold)
    if top_confidence < low_conf_threshold:
        return {
            "routed_specialty": GENERAL_MEDICINE_DEPT,
            "routed_confidence": top_confidence,
            "is_fallback": True,
            "fallback_reason": "low_confidence",
            "fallback_note": (
                f"Prediction confidence for the top specialty ({top_specialty}) was low ({top_confidence * 100:.1f}% < {low_conf_threshold * 100:.0f}%). "
                "We recommend starting with General Medicine, who can evaluate your overall condition and refer you onward."
            ),
            "possible_referrals": possible_referrals
        }

    # Condition 2: Ambiguity Margin (Top-1 - Top-2 < margin)
    margin = top_confidence - second_confidence
    if margin < ambiguity_margin:
        return {
            "routed_specialty": GENERAL_MEDICINE_DEPT,
            "routed_confidence": top_confidence,
            "is_fallback": True,
            "fallback_reason": "ambiguous",
            "fallback_note": (
                f"Symptoms match multiple departments closely ({top_specialty} vs. {second_specialty}, margin {margin * 100:.1f}% < {ambiguity_margin * 100:.0f}%). "
                "We recommend starting with General Medicine, who can perform an initial workup and refer you onward."
            ),
            "possible_referrals": possible_referrals
        }

    # Normal routing: High confidence, distinct specialist match
    image_only = (has_image_evidence and "Text" not in active_modalities and "Speech" not in active_modalities)
    normal_note = "Based on the photo only. Add a description for a more reliable result." if image_only else ""
    return {
        "routed_specialty": top_specialty,
        "routed_confidence": top_confidence,
        "is_fallback": False,
        "fallback_reason": "none",
        "fallback_note": normal_note,
        "possible_referrals": []
    }


def dynamic_multimodal_fusion(
    text_probs: Dict[str, float],
    speech_probs: Dict[str, float],
    image_probs: Dict[str, float],
    has_text: bool,
    has_speech: bool,
    has_image: bool,
    w1_text: float = 1.0,
    w2_speech: float = 1.0,
    w3_image: float = 1.0,
    specialties: Optional[List[str]] = None,
    unified_symptom_text: str = "",
    low_conf_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN
) -> Dict[str, Any]:
    """
    Computes dynamic late fusion across active modalities:
      S_final(c) = w_hat_text * P_text(c) + w_hat_speech * P_speech(c) + w_hat_image * P_image(c)

    Where w_hat_m is normalized across ONLY active user-supplied modalities.
    Subsequently runs the General Medicine Fallback Engine.
    """
    if specialties is None:
        specialties = (
            list(text_probs.keys())
            or list(speech_probs.keys())
            or list(image_probs.keys())
        )

    # A modality counts as active when its probability vector is non-zero (sum > 0)
    has_active_text = bool(has_text and sum(text_probs.values()) > 0)
    has_active_speech = bool(has_speech and sum(speech_probs.values()) > 0)
    has_active_image = bool(has_image and sum(image_probs.values()) > 0)

    i_text = float(has_active_text)
    i_speech = float(has_active_speech)
    i_image = float(has_active_image)

    # Active raw weights
    a_text = max(w1_text, 0.0) * i_text
    a_speech = max(w2_speech, 0.0) * i_speech
    a_image = max(w3_image, 0.0) * i_image
    w_sum = a_text + a_speech + a_image

    # Fallback to equal weighting if all active weights are 0
    if w_sum == 0.0:
        a_text = i_text
        a_speech = i_speech
        a_image = i_image
        w_sum = max(a_text + a_speech + a_image, 1.0)

    nw_text = a_text / w_sum
    nw_speech = a_speech / w_sum
    nw_image = a_image / w_sum

    active_modalities = []
    if has_active_text:
        active_modalities.append("Text")
    if has_active_speech:
        active_modalities.append("Speech")
    if has_active_image:
        active_modalities.append("Image")

    # If no inputs provided
    if not active_modalities:
        uniform_val = 1.0 / len(specialties) if specialties else 0.0
        uniform_probs = {spec: uniform_val for spec in specialties}
        sorted_specs = sorted(uniform_probs.items(), key=lambda x: x[1], reverse=True)
        fallback = apply_general_medicine_fallback(
            uniform_probs,
            unified_symptom_text="",
            active_modalities=[],
            low_conf_threshold=low_conf_threshold,
            ambiguity_margin=ambiguity_margin
        )
        return {
            "fused_probabilities": dict(sorted_specs),
            "top_specialty": sorted_specs[0][0] if sorted_specs else GENERAL_MEDICINE_DEPT,
            "top_confidence": sorted_specs[0][1] if sorted_specs else 0.0,
            "top_3_specialties": sorted_specs[:3],
            "normalized_weights": (0.0, 0.0, 0.0),
            "active_modalities": [],
            "fallback": fallback
        }

    # Dynamic Late Fusion
    fused_probs = {}
    for spec in specialties:
        p_t = text_probs.get(spec, 0.0)
        p_s = speech_probs.get(spec, 0.0)
        p_i = image_probs.get(spec, 0.0)
        fused_probs[spec] = (nw_text * p_t) + (nw_speech * p_s) + (nw_image * p_i)

    total_fused = sum(fused_probs.values()) or 1.0
    fused_probs = {s: float(v / total_fused) for s, v in fused_probs.items()}

    sorted_specs = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
    top_specialty, top_confidence = sorted_specs[0]
    top_3_specialties = sorted_specs[:3]

    # Evaluate General Medicine Fallback
    fallback = apply_general_medicine_fallback(
        fused_probabilities=fused_probs,
        unified_symptom_text=unified_symptom_text,
        active_modalities=active_modalities,
        low_conf_threshold=low_conf_threshold,
        ambiguity_margin=ambiguity_margin
    )

    return {
        "fused_probabilities": dict(sorted_specs),
        "top_specialty": top_specialty,
        "top_confidence": top_confidence,
        "top_3_specialties": top_3_specialties,
        "normalized_weights": (round(nw_text, 4), round(nw_speech, 4), round(nw_image, 4)),
        "active_modalities": active_modalities,
        "fallback": fallback
    }
