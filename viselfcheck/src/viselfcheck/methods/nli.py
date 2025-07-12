from typing import List, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.utils import seg_list_fn, get_word_segmentation_model
from ..config.settings import NLIConfig
from ..base import SelfCheckBase

class SelfCheckNLI(SelfCheckBase):
    """
    SelfCheckGPT (NLI variant): Checking LLM's text against its own sampled texts via DeBERTa-v3 finetuned to Multi-NLI
    """
    def __init__(
        self,
        nli_model: Optional[str] = None,
        device = None,
        do_word_segmentation = None
    ):
        self.nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.do_word_segmentation = do_word_segmentation if do_word_segmentation is not None else NLIConfig.do_word_segmentation

        self.model = AutoModelForSequenceClassification.from_pretrained(self.nli_model).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model)

        if self.do_word_segmentation:
            self.rdrsegmenter = get_word_segmentation_model()

        print("SelfCheck-NLI initialized to device", self.device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (0-1 range, higher means more inconsistent)
            Note: This is P(contradiction|sentence, sample) - we normalize the probability on 
            "entailment" or "contradiction" classes only and return the probability of the "contradiction" class
        """
        if self.do_word_segmentation:
            sentences = seg_list_fn(sentences, self.rdrsegmenter)
            sampled_passages = seg_list_fn(sampled_passages, self.rdrsegmenter)

        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, sample)],
                    add_special_tokens=True, padding='longest',
                    truncation=True, return_tensors='pt',
                    return_token_type_ids=True, return_attention_mask=True,
                    max_length=256
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits[:, [0, -1]] # neutral is already removed
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][-1].item() # prob(contradiction)
                scores[sent_i, sample_i] = prob_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence.tolist()