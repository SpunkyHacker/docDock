"""
Hybrid Medical Image Analyzer combining Local BiomedCLIP and Cloud Groq Vision (Llama-4-Scout).
Computes zero-shot ViT-B/16 visual features and synthesizes clinical visual descriptions
to achieve multimodal image specialty classification with robust offline fallback.
"""

import io
import base64
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from groq import Groq

from core.config import (
    BIOMEDCLIP_ID,
    GROQ_VISION_MODEL,
    IMAGE_CLIP_WEIGHT,
    IMAGE_VISION_TEXT_WEIGHT,
    IMAGE_TEMPERATURE,
    get_torch_device,
)
from core.data import VISUAL_SPECIALTY_PROMPTS
from core.text_engine import HybridMedicalTextClassifier


class HybridMedicalImageAnalyzer:
    """
    Part A (Local): microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    Part B (Cloud): meta-llama/llama-4-scout-17b-16e-instruct -> HybridMedicalTextClassifier
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.specialties = list(VISUAL_SPECIALTY_PROMPTS.keys())
        self.device = device or get_torch_device()

        # Load Local BiomedCLIP with compatibility across open_clip versions
        transforms_out = open_clip.create_model_and_transforms(BIOMEDCLIP_ID)
        if len(transforms_out) == 2:
            self.model, self.preprocess = transforms_out
        else:
            self.model = transforms_out[0]
            # Use validation transform (index 2) for evaluation/inference
            self.preprocess = transforms_out[2]

        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-tokenize all 8 visual specialty prompts once at startup
        self.prompt_tokens = self.tokenizer(
            list(VISUAL_SPECIALTY_PROMPTS.values()),
            context_length=256
        ).to(self.device)

    def _extract_biomedclip_scores(self, pil_image: Image.Image) -> Dict[str, float]:
        """
        Runs local BiomedCLIP forward pass to compute zero-shot visual probabilities.
        Handles variations in model forward return signature.
        """
        img_tensor = self.preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor, self.prompt_tokens)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                image_features, text_features, logit_scale = outputs[0], outputs[1], outputs[2]
            elif isinstance(outputs, dict):
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logit_scale = outputs.get("logit_scale", getattr(self.model, "logit_scale", torch.tensor(1.0)).exp())
            else:
                image_features = self.model.encode_image(img_tensor)
                text_features = self.model.encode_text(self.prompt_tokens)
                logit_scale = getattr(self.model, "logit_scale", torch.tensor(1.0)).exp()

            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Scale logits with IMAGE_TEMPERATURE to prevent over-saturation
            logits = (logit_scale * IMAGE_TEMPERATURE * image_features @ text_features.t()).squeeze(0)
            clip_probs = F.softmax(logits, dim=0).cpu().tolist()

        return dict(zip(self.specialties, clip_probs))

    def analyze_image(
        self,
        pil_image: Optional[Image.Image],
        hybrid_text_engine: HybridMedicalTextClassifier,
        groq_api_key: str = ""
    ) -> Dict[str, Any]:
        """
        Analyzes uploaded patient symptom photograph:
        1. Computes local BiomedCLIP zero-shot probabilities.
        2. Calls Groq Llama-4-Scout Vision to generate 1-sentence clinical description.
        3. Routes description into local HybridMedicalTextClassifier.
        4. Fuses 50% BiomedCLIP + 50% Visual-Text (or 100% BiomedCLIP if Vision API unavailable).
        """
        zero_probs = {spec: 0.0 for spec in self.specialties}

        if pil_image is None:
            return {
                "probs": zero_probs,
                "visual_description": "",
                "status": "Inactive (No image provided)",
                "error": None
            }

        # Step 1: Run Local BiomedCLIP
        try:
            biomedclip_scores = self._extract_biomedclip_scores(pil_image)
        except Exception as e:
            return {
                "probs": zero_probs,
                "visual_description": "",
                "status": f"BiomedCLIP processing error: {e}",
                "error": str(e)
            }

        # Step 2: Groq Vision-to-Text Description
        visual_description = ""
        visual_text_scores = {}
        clean_key = (groq_api_key or "").strip()

        if clean_key:
            try:
                buf = io.BytesIO()
                # Downscale if massive to conserve bandwidth and latency
                export_img = pil_image.convert("RGB")
                if max(export_img.size) > 1024:
                    export_img.thumbnail((1024, 1024))
                export_img.save(buf, format="JPEG", quality=85)
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                client = Groq(api_key=clean_key, timeout=20.0)
                vlm_resp = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    temperature=0.1,
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Describe the visible clinical symptoms, anatomical region, and likely "
                                        "medical condition in this patient image in 1 concise sentence "
                                        "(e.g., erythematous scaly skin rash on elbow, red inflamed eye, swollen knee joint)."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                }
                            ]
                        }
                    ]
                )
                if vlm_resp.choices and vlm_resp.choices[0].message.content:
                    visual_description = vlm_resp.choices[0].message.content.strip()
                    if visual_description:
                        visual_text_scores = hybrid_text_engine.predict_specialty(visual_description)
                    else:
                        print("[Image Engine Warning] Groq Vision returned empty description. Continuing with local BiomedCLIP only.")
                else:
                    print("[Image Engine Warning] Groq Vision returned no choices. Continuing with local BiomedCLIP only.")

            except Exception as e:
                # Non-fatal: log one-line warning without exposing secrets and degrade to BiomedCLIP
                print(f"[Image Engine Warning] Groq Vision unavailable ({type(e).__name__}). Continuing with local BiomedCLIP only.")

        # Step 3: Modality Combination
        if visual_text_scores:
            final_probs = {
                spec: (IMAGE_CLIP_WEIGHT * biomedclip_scores.get(spec, 0.0)) +
                      (IMAGE_VISION_TEXT_WEIGHT * visual_text_scores.get(spec, 0.0))
                for spec in self.specialties
            }
            # Renormalize sum to 1.0
            total = sum(final_probs.values()) or 1.0
            final_probs = {s: float(v / total) for s, v in final_probs.items()}
            status = "Local BiomedCLIP (50%) + Groq Llama-4-Scout -> Hybrid Text Engine (50%)"
        else:
            final_probs = biomedclip_scores
            status = "Local BiomedCLIP (100% - Local Mode / Vision Fallback)"

        return {
            "probs": final_probs,
            "visual_description": visual_description,
            "status": status,
            "error": None
        }
