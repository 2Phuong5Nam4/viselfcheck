from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class SelfCheckBase(ABC):
    """
    Base class for self-checking modules.
    """

    @abstractmethod
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
            :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
            :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
            :return sent_scores: sentence-level scores
        """
        pass