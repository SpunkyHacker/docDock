"""
Local Hybrid Clinical Text Classifier combining PubMedBERT Bi-Encoder and DeBERTa-v3 NLI Cross-Encoder.
Pre-computes clinical embeddings at initialization for sub-300ms inference.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline

from core.config import (
    BI_ENCODER_ID,
    CROSS_ENCODER_ID,
    BI_ENCODER_TEMPERATURE,
    TEXT_CROSS_WEIGHT,
    MAX_TEXT_WORDS,
    get_torch_device,
    get_pipeline_device,
)
from core.data import DEPARTMENT_SYMPTOM_PROFILES


class HybridMedicalTextClassifier:
    """
    Local Hybrid Medical Text Engine running on CPU or GPU/MPS.
    Shared by:
      1. Typed Patient Symptom Text (Modality 1)
      2. Spoken Speech Transcript from Whisper (Modality 2)
      3. Visual Symptom Clinical Description from Vision API (Modality 3)
    """

    def __init__(self, device: Optional[torch.device] = None, pipeline_device=None):
        self.specialties: List[str] = list(DEPARTMENT_SYMPTOM_PROFILES.keys())
        self.device = device or get_torch_device()
        
        # Determine pipeline device parameter
        if pipeline_device is None:
            pipeline_device = get_pipeline_device()

        # Load Part A: Local Medical Bi-Encoder (NeuML/pubmedbert-base-embeddings)
        self.bi_tokenizer = AutoTokenizer.from_pretrained(BI_ENCODER_ID)
        self.bi_model = AutoModel.from_pretrained(BI_ENCODER_ID).to(self.device)
        self.bi_model.eval()

        # Precompute 768-D embeddings for all 8 hospital departments once at startup
        self.dept_embeddings = self._encode_bi_encoder(
            list(DEPARTMENT_SYMPTOM_PROFILES.values())
        )

        # Load Part B: Local Zero-Shot NLI Cross-Encoder (MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
        # Attempt hardware acceleration, gracefully falling back to CPU if necessary
        try:
            self.cross_classifier = pipeline(
                "zero-shot-classification",
                model=CROSS_ENCODER_ID,
                device=pipeline_device,
            )
        except Exception as e:
            # Fallback to CPU pipeline if MPS or specific device index encounters issue
            print(f"[Text Engine] Fallback to CPU pipeline due to: {e}")
            self.cross_classifier = pipeline(
                "zero-shot-classification",
                model=CROSS_ENCODER_ID,
                device=-1,
            )

    def _encode_bi_encoder(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenizes and generates mean-pooled, L2-normalized embeddings using PubMedBERT.
        """
        inputs = self.bi_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bi_model(**inputs)
            token_embeddings = outputs[0]
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return F.normalize(mean_pooled, p=2, dim=1)

    def predict_specialty(
        self,
        symptom_text: str,
        cross_weight: float = TEXT_CROSS_WEIGHT
    ) -> Dict[str, float]:
        """
        Predicts department probabilities using late fusion between:
        1. PubMedBERT Bi-Encoder Cosine Similarity with temperature scaling (tau=12.0)
        2. DeBERTa-v3 Cross-Encoder Zero-Shot Entailment
        """
        if not symptom_text or not symptom_text.strip():
            return {spec: 0.0 for spec in self.specialties}

        # Cap text at MAX_TEXT_WORDS (350 words) to prevent exceeding tokenizer limits
        clean_text = " ".join(symptom_text.strip().split()[:MAX_TEXT_WORDS])

        # 1. Bi-Encoder Cosine Similarity Pass (~15ms)
        query_vec = self._encode_bi_encoder([clean_text])
        cos_sim = torch.matmul(query_vec, self.dept_embeddings.T).squeeze(0)
        bi_probs = F.softmax(cos_sim * BI_ENCODER_TEMPERATURE, dim=0).cpu().tolist()
        bi_scores = dict(zip(self.specialties, bi_probs))

        # 2. Cross-Encoder NLI Pass (~200ms)
        nli_out = self.cross_classifier(
            clean_text,
            candidate_labels=self.specialties,
            hypothesis_template="This patient describes clinical symptoms treated by the {} department.",
            multi_label=False,
            truncation=True
        )
        cross_scores = dict(zip(nli_out["labels"], nli_out["scores"]))

        # 3. Late Fusion: 55% Cross-Encoder + 45% Bi-Encoder
        bi_weight = 1.0 - cross_weight
        fused = {
            spec: (cross_weight * cross_scores.get(spec, 0.0)) + (bi_weight * bi_scores.get(spec, 0.0))
            for spec in self.specialties
        }
        total = sum(fused.values()) or 1.0
        return {spec: float(val / total) for spec, val in fused.items()}
