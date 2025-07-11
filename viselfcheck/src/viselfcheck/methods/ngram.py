from typing import List

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
        **kwargs
    ):
        
        smoothing_pseudo_count: int = kwargs.get('smoothing_pseudo_count', 0)

        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be integer >= 1")

        passage = " ".join(sentences)
        ngram_model.add(passage)

        for sampled_passge in sampled_passages:
            ngram_model.add(sampled_passge)

        ngram_model.train(k=smoothing_pseudo_count)
        ngram_pred = ngram_model.evaluate(sentences)

        # Extract sentence-level avg_neg_logprob scores and normalize them
        scores = ngram_pred['sent_level']['avg_neg_logprob']
        
        # Convert negative log probabilities to similarity scores (0-1 range)
        # Lower negative log probability = higher similarity
        # We'll use a simple transformation: 1 / (1 + score) for scores > 0
        normalized_scores = []
        for score in scores:
            if score <= 0:
                normalized_scores.append(1.0)  # Perfect match
            else:
                # Transform to 0-1 range where lower neg_logprob = higher similarity
                normalized_score = 1.0 / (1.0 + score)
                normalized_scores.append(min(1.0, max(0.0, normalized_score)))
        
        return normalized_scores