"""
docDock: Multimodal Framework for Specialist Doctor Recommendation
Gradio Interactive Web Application.

Integrates:
- Typed Text Symptoms (PubMedBERT + DeBERTa-v3 Hybrid Text Engine)
- Multilingual Spoken Voice (Groq Whisper-large-v3 Speech Translation)
- Clinical Photos (BiomedCLIP + Groq Llama-4-Scout Vision)
- Dynamic Weighted Late Fusion
- Negation-Aware Emergency Safety Triage
- 6-Factor Learning-to-Rank Specialist Doctor Recommendation
"""

import os
from typing import Tuple, Dict, Any, Optional
import gradio as gr
from PIL import Image
import pandas as pd

from core.config import (
    get_groq_api_key,
    BI_ENCODER_ID,
    CROSS_ENCODER_ID,
    BIOMEDCLIP_ID,
    GROQ_WHISPER_MODEL,
    GROQ_VISION_MODEL,
)
from core.text_engine import HybridMedicalTextClassifier
from core.voice_engine import process_voice_modality
from core.image_engine import HybridMedicalImageAnalyzer
from core.triage import evaluate_emergency_triage
from core.fusion import dynamic_multimodal_fusion
from core.ltr import rank_doctors

# ==============================================================================
# 1. INITIALIZE ENGINES ONCE AT STARTUP (SINGLETON PATTERN)
# ==============================================================================
print("[docDock Startup] Loading Local Hybrid Medical Text Classifier...")
GLOBAL_TEXT_ENGINE = HybridMedicalTextClassifier()

print("[docDock Startup] Loading Local BiomedCLIP Vision Engine...")
GLOBAL_IMAGE_ENGINE = HybridMedicalImageAnalyzer()

print("[docDock Startup] Engines initialized successfully.")


# ==============================================================================
# 2. CORE MULTIMODAL PIPELINE DISPATCHER
# ==============================================================================
def run_docdock_pipeline(
    typed_text: str,
    audio_filepath: Optional[str],
    pil_image: Optional[Image.Image],
    w1_text: float,
    w2_speech: float,
    w3_image: float,
    ui_groq_key: str
) -> Tuple[str, str, Dict[str, float], str, str, str, pd.DataFrame, str]:
    """
    Executes the 4-stage multimodal clinical routing pipeline:
      1. Ingestion across Text, Voice, and Image
      2. Feature Extraction & Probability Estimation
      3. Safety Triage & Dynamic Weighted Multimodal Fusion
      4. Specialist Doctor Learning-to-Rank (LTR) Recommendation
    """
    groq_key = get_groq_api_key(ui_groq_key)
    specialties = GLOBAL_TEXT_ENGINE.specialties

    # --------------------------------------------------------------------------
    # Stage 1: Modality Execution
    # --------------------------------------------------------------------------
    # Modality 1: Typed Text
    has_text = bool(typed_text and typed_text.strip())
    if has_text:
        text_probs = GLOBAL_TEXT_ENGINE.predict_specialty(typed_text)
    else:
        text_probs = {s: 0.0 for s in specialties}

    # Modality 2: Spoken Voice (Groq whisper-large-v3 -> Local Hybrid Text Engine)
    has_speech = bool(audio_filepath and os.path.exists(audio_filepath))
    if has_speech:
        voice_out = process_voice_modality(audio_filepath, groq_key, GLOBAL_TEXT_ENGINE)
        speech_transcript = voice_out.get("transcript", "")
        speech_probs = voice_out.get("probs", {s: 0.0 for s in specialties})
        speech_status = voice_out.get("status", "")
    else:
        speech_transcript = ""
        speech_probs = {s: 0.0 for s in specialties}
        speech_status = "Inactive (No audio input provided)"

    # Modality 3: Clinical Image (Local BiomedCLIP + Groq Vision -> Local Text Engine)
    has_image = (pil_image is not None)
    if has_image:
        image_out = GLOBAL_IMAGE_ENGINE.analyze_image(pil_image, GLOBAL_TEXT_ENGINE, groq_key)
        image_probs = image_out.get("probs", {s: 0.0 for s in specialties})
        visual_description = image_out.get("visual_description", "")
        image_status = image_out.get("status", "")
    else:
        image_probs = {s: 0.0 for s in specialties}
        visual_description = ""
        image_status = "Inactive (No image provided)"

    # --------------------------------------------------------------------------
    # Stage 2: Unified Symptom Corpus & Emergency Safety Triage
    # --------------------------------------------------------------------------
    parts = []
    if has_text:
        parts.append(typed_text.strip())
    if speech_transcript:
        parts.append(speech_transcript.strip())
    if visual_description:
        parts.append(visual_description.strip())
    unified_symptom_text = " ".join(parts).strip()

    triage_result = evaluate_emergency_triage(unified_symptom_text)
    triage_banner_html = triage_result["banner_html"]

    # --------------------------------------------------------------------------
    # Stage 3: Dynamic Weighted Multimodal Late Fusion & Fallback
    # --------------------------------------------------------------------------
    fusion_result = dynamic_multimodal_fusion(
        text_probs=text_probs,
        speech_probs=speech_probs,
        image_probs=image_probs,
        has_text=has_text,
        has_speech=has_speech,
        has_image=has_image,
        w1_text=w1_text,
        w2_speech=w2_speech,
        w3_image=w3_image,
        specialties=specialties,
        unified_symptom_text=unified_symptom_text
    )

    top_specialty = fusion_result["top_specialty"]
    top_confidence = fusion_result["top_confidence"]
    top_3_specialties = fusion_result["top_3_specialties"]
    nw1, nw2, nw3 = fusion_result["normalized_weights"]
    fused_probabilities = fusion_result["fused_probabilities"]
    fallback = fusion_result["fallback"]

    routed_specialty = fallback["routed_specialty"]
    routed_confidence = fallback["routed_confidence"]
    is_fallback = fallback["is_fallback"]
    fallback_reason = fallback["fallback_reason"]
    fallback_note = fallback["fallback_note"]
    possible_referrals = fallback["possible_referrals"]

    # Format routed specialty card and fallback referrals
    if is_fallback:
        referrals_str = ""
        if possible_referrals:
            ref_items = [f"• **{d}:** `{p*100:.1f}%`" for d, p in possible_referrals]
            referrals_str = "\n\n**Possible Specialist Referrals:**  \n" + " &nbsp;|&nbsp; ".join(ref_items)

        top_specialty_markdown = (
            f"### 🩺 Recommended Hospital Department\n"
            f"## **{routed_specialty}** *(Initial Triage)*\n\n"
            f"> ℹ️ **Routing Reason (`{fallback_reason}`):** {fallback_note}"
            f"{referrals_str}"
        )
    elif fallback_reason == "general_is_top":
        top_specialty_markdown = (
            f"### 🩺 Recommended Hospital Department\n"
            f"## **{routed_specialty}**\n"
            f"**Prediction Confidence:** `{routed_confidence * 100:.1f}%`  \n"
            f"> ℹ️ {fallback_note}"
        )
    else:
        note_str = f"  \n> ℹ️ {fallback_note}" if fallback_note else ""
        top_specialty_markdown = (
            f"### 🩺 Recommended Hospital Department\n"
            f"## **{routed_specialty}**\n"
            f"**Prediction Confidence:** `{routed_confidence * 100:.1f}%`{note_str}"
        )

    # Format top-3 bar chart / distribution
    top_3_dict = {dept: round(prob, 4) for dept, prob in top_3_specialties}

    # Format normalized weights display
    weights_display = (
        f"**Active Normalized Weights (∑ = 1.0):**  \n"
        f"• **Text (ŵ₁):** `{nw1:.2f}` &nbsp;|&nbsp; "
        f"• **Voice (ŵ₂):** `{nw2:.2f}` &nbsp;|&nbsp; "
        f"• **Image (ŵ₃):** `{nw3:.2f}`"
    )

    # --------------------------------------------------------------------------
    # Stage 4: Specialist Doctor Learning-to-Rank (LTR) Recommendation
    # --------------------------------------------------------------------------
    ref_depts = [r[0] for r in possible_referrals] if is_fallback else None
    ranked_doctors_df = rank_doctors(
        fused_probabilities=fused_probabilities,
        unified_symptom_text=unified_symptom_text,
        routed_specialty=routed_specialty,
        referral_specialties=ref_depts
    )

    # Pipeline audit status message
    audit_msg = (
        f"**Pipeline Diagnostics:**  \n"
        f"• Routed Department: **{routed_specialty}** (Fallback: `{fallback_reason}`)  \n"
        f"• Active Inputs: {', '.join(fusion_result['active_modalities']) if fusion_result['active_modalities'] else 'None'}  \n"
        f"• Speech Engine: {speech_status}  \n"
        f"• Vision Engine: {image_status}  \n"
        f"• Triage Status: {'🚨 RED FLAG DETECTED' if triage_result['is_emergency'] else '✅ Normal Outpatient'}"
    )

    return (
        triage_banner_html,
        top_specialty_markdown,
        top_3_dict,
        weights_display,
        speech_transcript or "(No speech transcript generated)",
        visual_description or "(No visual description generated)",
        ranked_doctors_df,
        audit_msg
    )


# ==============================================================================
# 3. GRADIO USER INTERFACE
# ==============================================================================
custom_css = """
.gradio-container { max-width: 1280px !important; margin: auto !important; }
.main-header { text-align: center; margin-bottom: 20px; }
.disclaimer-box {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    color: #92400e;
    font-size: 13.5px;
}
"""

with gr.Blocks(title="docDock: Multimodal Doctor Recommendation") as demo:
    # Header
    gr.Markdown(
        """
        # 🏥 docDock: Multimodal Framework for Specialist Doctor Recommendation
        ### *Multimodal Symptom Analysis across Text, Multilingual Voice, and Clinical Photography*
        **Stat-a-thon Course Project** — Multispecialty Hospital Outpatient Triage & Specialist Matching Engine
        """,
        elem_classes=["main-header"]
    )

    # Non-Dismissible Visible Medical Disclaimer (Gap 10)
    gr.Markdown(
        """
        <div class='disclaimer-box'>
        ⚠️ <strong>Mandatory Clinical Research Disclaimer:</strong><br/>
        docDock is an experimental administrative decision-support research prototype modeled after multispecialty hospital triage protocols (e.g. CMC Vellore). 
        It is designed strictly for research demonstration and is <strong>NOT</strong> a certified medical diagnostic device. It does not provide medical diagnosis, clinical evaluation, or treatment prescriptions.
        If you are experiencing severe, sudden, or life-threatening symptoms, immediately proceed to the nearest Emergency Department (Casualty) or dial emergency services (108 in India / 911).
        </div>
        """
    )

    # Dynamic Emergency Safety Banner (Gap 1)
    emergency_banner_output = gr.HTML(
        value=(
            "<div style='background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; "
            "padding: 10px 14px; margin-bottom: 12px; color: #166534; font-size: 13px;'>"
            "✅ <strong>Safety Triage Ready:</strong> Ready for symptom ingestion."
            "</div>"
        )
    )

    with gr.Row():
        # LEFT COLUMN: Patient Symptom Ingestion
        with gr.Column(scale=5):
            gr.Markdown("### 📥 1. Patient Symptom Ingestion")

            input_text = gr.Textbox(
                label="Typed Symptom Description (Modality 1: Text)",
                placeholder="Type symptoms in plain English (e.g., 'itchy red patches on elbow' or 'acid reflux after spicy meals, no chest pain')...",
                lines=3
            )

            input_audio = gr.Audio(
                label="Spoken Symptom Audio (Modality 2: Spoken Voice - Tamil / Hindi / Tanglish / Hinglish / English)",
                sources=["upload", "microphone"],
                type="filepath"
            )

            input_image = gr.Image(
                label="Clinical Symptom Photograph (Modality 3: Vision - Skin / Eye / Joint / Wound)",
                type="pil"
            )

            with gr.Accordion("⚙️ Advanced Multimodal Fusion Weights & Cloud API Settings", open=False):
                gr.Markdown("Adjust initial importance weights ($w_1, w_2, w_3$). Missing modalities are automatically zeroed out and renormalised.")
                slider_w1 = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Text Weight (w1)")
                slider_w2 = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Speech Weight (w2)")
                slider_w3 = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Image Weight (w3)")
                
                input_api_key = gr.Textbox(
                    label="Groq API Key (Optional Override)",
                    type="password",
                    placeholder="Leave empty to use GROQ_API_KEY from .env"
                )

            with gr.Row():
                btn_submit = gr.Button("🔍 Analyze & Recommend Specialist", variant="primary", scale=2)
                btn_clear = gr.Button("🔄 Clear Inputs", variant="secondary", scale=1)

        # RIGHT COLUMN: Clinical Routing & Doctor Recommendations
        with gr.Column(scale=6):
            gr.Markdown("### 📊 2. Clinical Routing & Specialist Recommendations")

            output_specialty_card = gr.Markdown(
                "### 🩺 Recommended Hospital Department\n*Submit patient symptoms on the left to view routing results.*"
            )

            output_top3_chart = gr.Label(
                num_top_classes=3,
                label="Top-3 Hospital Departments (Posterior Probability Distribution)"
            )

            output_weights_display = gr.Markdown(
                "**Active Normalized Weights (∑ = 1.0):** Inactive"
            )

            with gr.Accordion("📝 Extracted Multimodal Clinical Features", open=True):
                output_transcript = gr.Textbox(
                    label="Groq Whisper-large-v3 Translated English Transcript",
                    interactive=False,
                    lines=2
                )
                output_visual_desc = gr.Textbox(
                    label="Groq Llama-4-Scout Clinical Visual Description",
                    interactive=False,
                    lines=2
                )

            gr.Markdown("### 👨‍⚕️ Ranked Specialist Physicians")
            output_doctor_table = gr.DataFrame(
                label="Matched Specialist Doctors (Ranked by 6-Factor Learning-to-Rank Formula)",
                interactive=False,
                wrap=True
            )

            output_audit = gr.Markdown("**Pipeline Diagnostics:** Idle")

    # Hook Submit & Clear Actions
    pipeline_inputs = [
        input_text,
        input_audio,
        input_image,
        slider_w1,
        slider_w2,
        slider_w3,
        input_api_key
    ]

    pipeline_outputs = [
        emergency_banner_output,
        output_specialty_card,
        output_top3_chart,
        output_weights_display,
        output_transcript,
        output_visual_desc,
        output_doctor_table,
        output_audit
    ]

    btn_submit.click(
        fn=run_docdock_pipeline,
        inputs=pipeline_inputs,
        outputs=pipeline_outputs
    )

    def clear_all():
        return (
            "", None, None, 1.0, 1.0, 1.0, "",
            "<div style='background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #166534; font-size: 13px;'>✅ <strong>Safety Triage Ready:</strong> Ready for symptom ingestion.</div>",
            "### 🩺 Recommended Hospital Department\n*Submit patient symptoms on the left to view routing results.*",
            {},
            "**Active Normalized Weights (∑ = 1.0):** Inactive",
            "",
            "",
            pd.DataFrame(),
            "**Pipeline Diagnostics:** Idle"
        )

    btn_clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            input_text,
            input_audio,
            input_image,
            slider_w1,
            slider_w2,
            slider_w3,
            input_api_key
        ] + pipeline_outputs
    )

    # Pre-Loaded Blueprint Demo Cases (Blueprint Section 6)
    gr.Markdown("---")
    gr.Markdown("### 🧪 Quick Verification Checklist: Pre-Loaded Blueprint Demo Cases")
    gr.Examples(
        examples=[
            [
                "",
                "sample_data/tanglish_dermatology_symptom.wav",
                None,
                0.0, 1.0, 0.0, ""
            ],
            [
                "My stomach burns after eating spicy food, I have acid reflux, no chest pain.",
                None,
                None,
                1.0, 0.0, 0.0, ""
            ],
            [
                "I have an itchy red rash and inflamed patches on my skin.",
                None,
                "sample_data/skin_rash.jpg",
                1.0, 0.0, 1.0, ""
            ],
            [
                "I have fever, body ache and feel very tired for 3 days.",
                None,
                None,
                1.0, 0.0, 0.0, ""
            ],
            [
                "I don't feel well.",
                None,
                None,
                1.0, 0.0, 0.0, ""
            ],
            [
                "Crushing chest pain spreading to my left arm and jaw.",
                None,
                None,
                1.0, 0.0, 0.0, ""
            ]
        ],
        inputs=pipeline_inputs,
        outputs=pipeline_outputs,
        fn=run_docdock_pipeline,
        cache_examples=False,
        label="Click any demo case below to instantly test end-to-end functionality:"
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"[docDock] Launching Gradio Web App on http://{host}:{port}...")
    demo.launch(server_name=host, server_port=port, share=False)
