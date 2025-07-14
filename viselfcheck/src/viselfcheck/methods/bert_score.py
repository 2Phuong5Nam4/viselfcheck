from ..base import SelfCheckBase
from ..utils.utils import expand_list1, expand_list2
from ..config.settings import BertScoreConfig

from typing import List, Optional
import numpy as np
import torch
import bert_score
from underthesea import sent_tokenize



class SelfCheckBERTScore(SelfCheckBase):
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self,  rescale_with_baseline=False, device=None, lang: Optional[str] = None, model_type: Optional[str] = None, sent_tokenize=sent_tokenize):
        """
        :default_model: model for BERTScore
        :rescale_with_baseline:
            - whether or not to rescale the score. If False, the values of BERTScore will be very high
            - this issue was observed and later added to the BERTScore package,
            - see https://github.com/Tiiiger/bert_score/blob/master/journal/rThe inter-annotator agreementescale_baseline.md
        :model_type: bert specification, default using the suggested model for the target language
        """
        self.sent_tokenize = sent_tokenize
        
        # Priority logic: model_type > lang > default from config
        if model_type is not None:
            self.model_type = model_type
            self.lang = None  # Can be None when model_type is specified
        elif lang is not None:
            self.model_type = None
            self.lang = lang
        else:
            # Both are None, use default from config
            self.model_type = BertScoreConfig.model_type
            self.lang = None  # Default lang if config doesn't specify
            
        self.rescale_with_baseline = rescale_with_baseline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
            
        print("SelfCheck-BERTScore initialized with device:", self.device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        passage: Optional[str] = None,
        **kwargs
    )-> List[float]:
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            passage: Optional passage text. If provided, this will be used instead of joining sentences.
                    If None, will use " ".join(sentences) to create the passage.
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (0-1 range, higher means more inconsistent)
            Note: This is computed as 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.sent_tokenize(sample_passage)] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            # Priority: model_type > lang > error if both None
            if self.model_type is not None:
                # Use predefined model_type
                P, R, F1 = bert_score.score(
                        cands, refs,
                        verbose=False,
                        rescale_with_baseline=self.rescale_with_baseline,
                        device=self.device,
                        model_type=self.model_type,
                )
            elif self.lang is not None:
                # Use lang when model_type is None
                P, R, F1 = bert_score.score(
                        cands, refs,
                        verbose=False,
                        rescale_with_baseline=self.rescale_with_baseline,
                        device=self.device,
                        lang=self.lang,
                )
            else:
                # Both model_type and lang are None
                raise ValueError("Either 'model_type' or 'lang' must be specified for BERTScore computation")

            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(dim=1).values.cpu().numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent.tolist()  # Convert to list
