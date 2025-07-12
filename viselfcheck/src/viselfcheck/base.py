from abc import ABC, abstractmethod
from typing import List, Optional

class SelfCheckBase(ABC):
    """
    Base class for self-checking modules.
    """

    @abstractmethod
    def predict(self, sentences: List[str], sampled_passages: List[str], passage: Optional[str] = None, **kwargs) -> List[float]:
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            passage: Optional passage text. If provided, this will be used instead of joining sentences.
                    If None, will use " ".join(sentences) to create the passage.
            **kwargs: Additional method-specific parameters (see individual implementations)
            
        Returns:
            List of sentence-level scores (typically 0.0-1.0, higher means more inconsistent/hallucinated)
        """
        pass