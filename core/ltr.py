"""
Learning-to-Rank (LTR) Doctor Ranking Engine.
Ranks specialist physicians within the top predicted hospital department using a 6-factor
formula incorporating specialty confidence, stemmed sub-focus tag overlap, clinical success rate,
patient ratings, clinical experience, and real-time scheduling availability.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from core.config import (
    LTR_W_CONF,
    LTR_W_TAG,
    LTR_W_SUCCESS,
    LTR_W_RATING,
    LTR_W_EXP,
    LTR_W_AVAIL,
    LTR_EXP_MAX_YEARS,
)
from core.data import DOCTORS_DB


def _simple_stem(token: str) -> str:
    """
    Lightweight rule-based clinical suffix stemmer without requiring heavy NLTK downloads.
    Handles common medical plurals and gerunds (e.g., 'itching' -> 'itch', 'patches' -> 'patch').
    """
    t = token.lower().strip()
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 4:
        # e.g., 'itching' -> 'itch', 'swelling' -> 'swell'
        stem = t[:-3]
        if stem.endswith("ll") or stem.endswith("ss"):
            return stem
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem
    if t.endswith("es") and len(t) > 3 and (t.endswith("shes") or t.endswith("ches") or t.endswith("sses")):
        return t[:-2]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _tokenize_and_stem(text: str) -> set[str]:
    """
    Extracts alphanumeric tokens and their stemmed representations.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stems = set(tokens)
    for tok in tokens:
        stems.add(_simple_stem(tok))
    return stems


def calculate_tag_overlap(sub_focus: str, symptom_corpus: str) -> Tuple[float, List[str]]:
    """
    Computes stemmed token and phrase overlap between doctor's sub-focus tags and
    the patient's multimodal symptom description.

    Returns:
        overlap_score: float in [0.0, 1.0]
        matched_terms: list of matched sub-focus tags
    """
    if not sub_focus or not symptom_corpus:
        return 0.0, []

    corpus_lower = symptom_corpus.lower()
    corpus_stems = _tokenize_and_stem(corpus_lower)

    tag_list = [t.strip() for t in sub_focus.split(",") if t.strip()]
    matched_tags = []

    for tag in tag_list:
        tag_lower = tag.lower()
        # Direct phrase match
        if tag_lower in corpus_lower:
            matched_tags.append(tag)
            continue

        # Stemmed word overlap match (if multi-word or single word)
        tag_tokens = re.findall(r"[a-zA-Z0-9]+", tag_lower)
        if not tag_tokens:
            continue

        tag_stems = [_simple_stem(tok) for tok in tag_tokens]
        # Check if all tokens in the sub-focus tag match stems in the corpus
        if all(s in corpus_stems for s in tag_stems):
            matched_tags.append(tag)

    # Overlap score: 2 or more matched terms yields 1.0; strictly capped at 1.0
    overlap_score = min(len(matched_tags) / 2.0, 1.0)
    return overlap_score, matched_tags


def rank_doctors(
    fused_probabilities: Dict[str, float],
    unified_symptom_text: str,
    doctors_db: Optional[List[Dict[str, Any]]] = None,
    include_secondary: bool = True,
    routed_specialty: Optional[str] = None,
    referral_specialties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Ranks specialist physicians according to the 6-factor LTR formula:
      R_d = 0.30 * S_final(c_d)
          + 0.25 * E_tag(d, Q_all)
          + 0.15 * success_rate_d
          + 0.10 * (rating_d / 5.0)
          + 0.10 * min(exp_years_d / 25.0, 1.0)
          + 0.10 * avail_score_d

    Prioritizes the routed specialty (e.g., General Medicine if fallback triggered,
    or the top specialist department otherwise).
    """
    db = doctors_db if doctors_db is not None else DOCTORS_DB
    if not fused_probabilities or not db:
        return pd.DataFrame()

    sorted_specs = sorted(fused_probabilities.items(), key=lambda x: x[1], reverse=True)
    primary_specialty = routed_specialty or sorted_specs[0][0]

    # Build set of eligible specialties to display
    eligible_specialties = {primary_specialty}
    if referral_specialties:
        for ref in referral_specialties:
            eligible_specialties.add(ref)
    elif include_secondary and len(sorted_specs) > 1:
        # Include top secondary specialty if different from primary
        for spec, _ in sorted_specs:
            if spec != primary_specialty:
                eligible_specialties.add(spec)
                break

    ranked_records = []
    for doc in db:
        doc_specialty = doc.get("specialty", "")
        if doc_specialty not in eligible_specialties:
            continue

        spec_conf = fused_probabilities.get(doc_specialty, 0.0)
        sub_focus = doc.get("sub_focus", "")
        tag_score, matched_terms = calculate_tag_overlap(sub_focus, unified_symptom_text)

        success_rate = float(doc.get("success_rate", 0.0))
        rating = float(doc.get("rating", 0.0))
        exp_years = float(doc.get("exp_years", 0.0))
        avail_score = float(doc.get("avail_score", 0.0))

        # 6-Factor Learning-to-Rank Formula
        ltr_raw = (
            (LTR_W_CONF * spec_conf) +
            (LTR_W_TAG * tag_score) +
            (LTR_W_SUCCESS * success_rate) +
            (LTR_W_RATING * (rating / 5.0)) +
            (LTR_W_EXP * min(exp_years / LTR_EXP_MAX_YEARS, 1.0)) +
            (LTR_W_AVAIL * avail_score)
        )

        ltr_score = round(max(0.0, min(100.0, ltr_raw * 100.0)), 1)
        is_primary = (doc_specialty == primary_specialty)
        matched_str = ", ".join(matched_terms) if matched_terms else sub_focus

        ranked_records.append({
            "is_primary": is_primary,
            "Doctor Name": doc["name"],
            "Department": doc_specialty,
            "Matched Sub-Focus": matched_str,
            "Experience": f"{int(exp_years)} yrs",
            "Clinical Record": f"{success_rate * 100:.1f}% ({doc.get('patients_treated', 0)} cases)",
            "Rating": f"⭐ {rating:.1f}/5.0",
            "Availability": doc.get("availability", "On Schedule"),
            "LTR Score": ltr_score
        })

    # Sort: Primary routed specialty first, then descending by LTR Score
    ranked_records.sort(key=lambda r: (r["is_primary"], r["LTR Score"]), reverse=True)

    df = pd.DataFrame(ranked_records)
    if "is_primary" in df.columns:
        df = df.drop(columns=["is_primary"])

    return df
