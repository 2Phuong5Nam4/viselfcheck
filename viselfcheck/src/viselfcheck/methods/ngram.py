from typing import List, Optional

from ..modeling.modeling_ngram import UnigramModel, NgramModel
from ..base import SelfCheckBase

class SelfCheckNgram(SelfCheckBase):
    """
    SelfCheckGPT (Ngram variant): Checking LLM's text against its own sampled texts via ngram model
    Note that this variant of SelfCheck score is not bounded in [0.0, 1.0]
    """
    def __init__(self, n: int, lowercase: bool = True):
        """
        :param n: n-gram model, n=1 is Unigram, n=2 is Bigram, etc.
        :param lowercase: whether or not to lowercase when counting n-grams
        """
        self.n = n
        self.lowercase = lowercase
        print(f"SelfCheck-{n}gram initialized")

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        passage: Optional[str] = None,
        smoothing_pseudo_count: Optional[int] = None,
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            passage: Optional passage text. If provided, this will be used instead of joining sentences.
                    If None, will use " ".join(sentences) to create the passage.
            smoothing_pseudo_count: Pseudo count for smoothing (default: 0)
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (0-1 range, higher means more inconsistent)
        """
        if smoothing_pseudo_count is None:
            smoothing_pseudo_count = 0

        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be integer >= 1")

        # Use provided passage or join sentences to form a passage
        if passage is not None:
            main_passage = passage
        else:
            main_passage = " ".join(sentences)
        
        ngram_model.add(main_passage)

        for sampled_passge in sampled_passages:
            ngram_model.add(sampled_passge)

        ngram_model.train(k=smoothing_pseudo_count)
        ngram_pred = ngram_model.evaluate(sentences)

        # Extract sentence-level avg_neg_logprob scores and normalize them
        scores = ngram_pred['sent_level']['avg_neg_logprob']
        

        
        return scores