"""
Clinical Emergency Safety Triage Engine.
Detects life-threatening red-flag symptoms across patient inputs with negation awareness
(e.g., 'no chest pain' or 'denies shortness of breath' will NOT trigger an emergency alert).
"""

import re
from typing import Dict, List, Tuple, Any

# Clinical Red-Flag Symptom Categories & Patterns
RED_FLAG_PATTERNS = {
    "Acute Coronary Syndrome / Cardiac Emergency": [
        r"\b(?:crushing|severe|radiating|pressure)\s+chest\s+pain\b",
        r"\bchest\s+pain\s+(?:radiating|spreading)\s+to\s+(?:my\s+)?(?:left\s+arm|jaw|neck|back)\b",
        r"\bleft\s+arm\s+(?:and|with)\s+chest\s+pain\b",
        r"\bheart\s+attack\b",
        r"\bacute\s+angina\b"
    ],
    "Severe Respiratory Distress": [
        r"\b(?:severe\s+)?difficulty\s+breathing\b",
        r"\b(?:severe\s+)?shortness\s+of\s+breath\b",
        r"\bgasping\s+for\s+(?:air|breath)\b",
        r"\bcan(?:not|'t)\s+breathe\b",
        r"\bstridor\b",
        r"\basphyxiation\b",
        r"\bsuffocation\b"
    ],
    "Acute Neurological / Stroke (FAST Signs)": [
        r"\bslurred\s+speech\b",
        r"\b(?:facial|face)\s+(?:droop|asymmetry|drooping)\b",
        r"\bsudden\s+(?:weakness|numbness|paralysis)\s+(?:on\s+one\s+side|in\s+arm|in\s+leg|in\s+face)?\b",
        r"\bacute\s+stroke\b",
        r"\bloss\s+of\s+(?:speech|vision\s+in\s+one\s+eye)\b"
    ],
    "Seizures & Convulsions": [
        r"\b(?:active\s+)?seizure(?:s)?\b",
        r"\bconvulsion(?:s)?\b",
        r"\bstatus\s+epilepticus\b",
        r"\bepileptic\s+fit(?:s)?\b"
    ],
    "Severe Hemorrhage / Active Bleeding": [
        r"\b(?:heavy|uncontrolled|profuse|massive)\s+bleeding\b",
        r"\bhemorrhage\b",
        r"\bcoughing\s+(?:up\s+)?blood\b",
        r"\bvomiting\s+blood\b",
        r"\bhematemesis\b",
        r"\bhemoptysis\b"
    ],
    "Loss of Consciousness / Syncope": [
        r"\bloss\s+of\s+consciousness\b",
        r"\bpassed\s+out\b",
        r"\bunresponsive\b",
        r"\bsyncope\b",
        r"\bcollapse\b"
    ],
    "Anaphylaxis / Airway Compromise": [
        r"\banaphylaxis\b",
        r"\bsevere\s+allergic\s+reaction\b",
        r"\bthroat\s+(?:swelling|closing)\b",
        r"\btongue\s+swelling\b"
    ]
}

# Negation Triggers (medical negation cues)
NEGATION_PREFIXES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bdenies\b",
    r"\bdenying\b",
    r"\bwithout\b",
    r"\brule\s+out\b",
    r"\bruled\s+out\b",
    r"\bnegative\s+for\b",
    r"\bzero\b",
    r"\bfree\s+of\b",
    r"\bneither\b",
    r"\bnone\b"
]


def _is_negated(clause_text: str, match_start: int) -> bool:
    """
    Checks if a matched pattern within a clause is preceded by a negation trigger
    within a 5-word lookback window without an intervening contrastive word.
    """
    preceding_text = clause_text[:match_start].strip()
    words = preceding_text.split()
    lookback_window = " ".join(words[-5:]).lower() if words else ""

    for neg in NEGATION_PREFIXES:
        if re.search(neg, lookback_window):
            # If there is a contrastive conjunction like 'but' after the negation, it might not negate
            # e.g., 'not headache but chest pain'
            contrast_match = re.search(r"\b(?:but|however|except)\b", lookback_window)
            if contrast_match:
                neg_match = re.search(neg, lookback_window)
                if neg_match and neg_match.start() < contrast_match.start():
                    # The negation happened BEFORE the contrastive word, so the post-contrast phrase is not negated
                    continue
            return True
    return False


def evaluate_emergency_triage(symptom_corpus: str) -> Dict[str, Any]:
    """
    Evaluates unified symptom text across all modalities for urgent red-flag conditions.
    Handles negation to ensure phrases like 'no chest pain' do not trigger false positives.

    Returns:
        dict containing:
            - is_emergency: bool
            - flagged_reasons: list of triggered categories and matches
            - alert_message: human-readable emergency recommendation
            - banner_html: styled HTML alert for UI display
    """
    if not symptom_corpus or not symptom_corpus.strip():
        return {
            "is_emergency": False,
            "flagged_reasons": [],
            "alert_message": "",
            "banner_html": ""
        }

    # Split into clauses by common sentence and phrase delimiters
    # e.g., '.', ';', ',', 'and', 'but', 'or'
    raw_clauses = re.split(r"[\.\;\n\,]|\b(?:but|however|although|whereas)\b", symptom_corpus.lower())
    
    flagged_reasons: List[str] = []

    for clause in raw_clauses:
        clause_clean = clause.strip()
        if not clause_clean:
            continue

        for category, patterns in RED_FLAG_PATTERNS.items():
            for pat in patterns:
                for match in re.finditer(pat, clause_clean):
                    if not _is_negated(clause_clean, match.start()):
                        matched_str = match.group(0)
                        entry = f"{category} ('{matched_str}')"
                        if entry not in flagged_reasons:
                            flagged_reasons.append(entry)

    is_emergency = len(flagged_reasons) > 0

    if is_emergency:
        alert_message = (
            "EMERGENCY WARNING: Potential life-threatening red-flag symptoms detected: "
            + "; ".join(flagged_reasons)
            + ". Proceed IMMEDIATELY to the nearest Hospital Emergency Department (Casualty) "
            "or call Emergency Medical Services (108 in India / 911) without delay."
        )
        banner_html = (
            "<div style='background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 8px; "
            "padding: 16px; margin-bottom: 16px; color: #991b1b;'>"
            "<h3 style='margin: 0 0 8px 0; color: #b91c1c; display: flex; align-items: center;'>"
            "🚨 CRITICAL EMERGENCY TRIAGE ALERT</h3>"
            f"<p style='margin: 0 0 8px 0; font-size: 15px; font-weight: 600;'>{alert_message}</p>"
            "<p style='margin: 0; font-size: 13px; color: #7f1d1d;'>"
            "Note: Department routing and outpatient doctor ranking below are provided for administrative reference only. "
            "Do NOT delay seeking urgent medical care.</p>"
            "</div>"
        )
    else:
        alert_message = "No emergency red-flag indicators detected. Proceeding with outpatient specialty routing."
        banner_html = (
            "<div style='background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; "
            "padding: 10px 14px; margin-bottom: 12px; color: #166534; font-size: 13px;'>"
            "✅ <strong>Safety Triage Passed:</strong> No acute life-threatening red-flag symptoms detected."
            "</div>"
        )

    return {
        "is_emergency": is_emergency,
        "flagged_reasons": flagged_reasons,
        "alert_message": alert_message,
        "banner_html": banner_html
    }
