"""
Unit tests for Emergency Triage Engine and Negation Logic.
"""

from core.triage import evaluate_emergency_triage


def test_triage_red_flag_triggered():
    """
    Test Case: 'crushing chest pain spreading to my left arm' -> Emergency Triggered
    """
    corpus = "crushing chest pain spreading to my left arm"
    result = evaluate_emergency_triage(corpus)
    
    assert result["is_emergency"] is True
    assert len(result["flagged_reasons"]) > 0
    assert "Acute Coronary Syndrome" in result["flagged_reasons"][0]
    assert "🚨 CRITICAL EMERGENCY TRIAGE ALERT" in result["banner_html"]


def test_triage_negation_not_triggered():
    """
    Test Case: 'stomach burns after spicy food, acid reflux, no chest pain' -> Negated, Not Triggered
    """
    corpus = "stomach burns after spicy food, acid reflux, no chest pain"
    result = evaluate_emergency_triage(corpus)
    
    assert result["is_emergency"] is False
    assert len(result["flagged_reasons"]) == 0
    assert "Safety Triage Passed" in result["banner_html"]


def test_triage_stroke_fast_triggered():
    """
    Test Case: Sudden FAST stroke symptoms -> Emergency Triggered
    """
    corpus = "patient woke up with slurred speech and facial droop on the left side"
    result = evaluate_emergency_triage(corpus)
    
    assert result["is_emergency"] is True
    reasons_str = " ".join(result["flagged_reasons"])
    assert "Stroke" in reasons_str or "Neurological" in reasons_str


def test_triage_clinical_denials_not_triggered():
    """
    Test Case: Multiple clinical negation phrases ('denies', 'without', 'never')
    """
    corpus = "routine checkup, patient denies shortness of breath, without bleeding, and has never had seizures"
    result = evaluate_emergency_triage(corpus)
    
    assert result["is_emergency"] is False
    assert len(result["flagged_reasons"]) == 0
