"""
Unit tests for Learning-to-Rank (LTR) Doctor Ranking Engine.
"""

from core.ltr import calculate_tag_overlap, rank_doctors


def test_ltr_scores_in_range_and_sorted_descending():
    """
    Test Case: All doctor LTR scores must be in [0.0, 100.0] and sorted descending.
    """
    fused_probs = {
        "General Dermatology & Cosmetology": 0.70,
        "Cardiology & Cardiothoracic": 0.20,
        "Orthopaedics & Rheumatology": 0.10
    }
    corpus = "itching, red patches, skin rash, dermatitis"

    df = rank_doctors(fused_probs, corpus)
    assert not df.empty

    scores = df["LTR Score"].tolist()
    # Check bounded in [0, 100]
    for s in scores:
        assert 0.0 <= s <= 100.0

    # Check sorted descending
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_tag_overlap_stemming_and_cap():
    """
    Test Case: Tag overlap handles stemming (e.g., 'itching' -> 'itch') and is strictly capped at 1.0.
    """
    sub_focus = "eczema, psoriasis, itch, rash, red patches, blisters"
    # Even if 5 terms match, score must be capped at 1.0
    corpus = "patient has severe itching, multiple red patches, active rash, and large blisters"
    
    score, matched = calculate_tag_overlap(sub_focus, corpus)
    assert score == 1.0
    assert len(matched) >= 2


def test_ltr_guarded_against_single_department():
    """
    Test Case: Ranking engine operates safely when only 1 specialty is supplied (no index out of bounds).
    """
    fused_probs = {"Cardiology & Cardiothoracic": 1.0}
    df = rank_doctors(fused_probs, "chest pain", include_secondary=True)
    
    assert not df.empty
    assert (df["Department"] == "Cardiology & Cardiothoracic").all()


def test_ltr_general_medicine_doctors_when_routed():
    """
    Test Case: When routed to General Medicine (Internal Medicine), General Medicine doctors
    are ranked primarily at the top of the recommendation table.
    """
    from core.config import GENERAL_MEDICINE_DEPT
    fused_probs = {
        "General Dermatology & Cosmetology": 0.38,
        "ENT (Otorhinolaryngology)": 0.35,
        GENERAL_MEDICINE_DEPT: 0.27
    }
    corpus = "I have mild fever and fatigue and feeling unwell"
    df = rank_doctors(
        fused_probabilities=fused_probs,
        unified_symptom_text=corpus,
        routed_specialty=GENERAL_MEDICINE_DEPT,
        referral_specialties=["General Dermatology & Cosmetology", "ENT (Otorhinolaryngology)"]
    )
    assert not df.empty
    # General Medicine doctor must be #1
    assert df.iloc[0]["Department"] == GENERAL_MEDICINE_DEPT
    assert (df["LTR Score"] >= 0.0).all() and (df["LTR Score"] <= 100.0).all()
