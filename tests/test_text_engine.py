"""
Unit tests for the Hybrid Medical Text Classifier.
Verifies semantic routing and negation understanding.
"""

def test_text_only_dermatology(text_engine):
    """
    Test Case 1: 'itchy red patches on my elbow' -> General Dermatology & Cosmetology
    """
    from core.fusion import dynamic_multimodal_fusion
    query = "itchy red patches on my elbow"
    probs = text_engine.predict_specialty(query)
    
    assert len(probs) == 9
    top_dept = max(probs.items(), key=lambda x: x[1])[0]
    assert top_dept == "General Dermatology & Cosmetology"
    assert probs["General Dermatology & Cosmetology"] > 0.35

    # Also test fallback_reason == 'none'
    fusion = dynamic_multimodal_fusion(
        probs, probs, probs, True, False, False, unified_symptom_text=query
    )
    assert fusion["fallback"]["fallback_reason"] == "none"
    assert fusion["fallback"]["routed_specialty"] == "General Dermatology & Cosmetology"


def test_text_negation_gastroenterology(text_engine):
    """
    Test Case 2: 'stomach burns after spicy food, acid reflux, no chest pain'
    -> Gastroenterology & Hepatology, with Cardiology suppressed by DeBERTa NLI.
    """
    query = "stomach burns after spicy food, acid reflux, no chest pain"
    probs = text_engine.predict_specialty(query)
    
    top_dept = max(probs.items(), key=lambda x: x[1])[0]
    assert top_dept == "Gastroenterology & Hepatology"
    assert probs["Gastroenterology & Hepatology"] > probs.get("Cardiology & Cardiothoracic", 0.0)
    assert probs["Gastroenterology & Hepatology"] > 0.40


def test_text_general_medicine_fever(text_engine):
    """
    Test Case 3: 'I have fever, body ache and feel very tired for 3 days'
    -> General Medicine (Internal Medicine)
    """
    query = "I have fever, body ache and feel very tired for 3 days"
    probs = text_engine.predict_specialty(query)
    
    top_dept = max(probs.items(), key=lambda x: x[1])[0]
    assert top_dept == "General Medicine (Internal Medicine)"
    assert probs["General Medicine (Internal Medicine)"] > 0.35
