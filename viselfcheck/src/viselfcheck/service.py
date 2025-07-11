from typing import List, Optional, Union, Dict, Any
import warnings

from .base import SelfCheckBase
from .methods.bert_score import SelfCheckBERTScore
from .methods.nli import SelfCheckNLI
from .methods.mqag import SelfCheckMQAG
from .methods.ngram import SelfCheckNgram
from .methods.prompt import SelfCheckAPIPrompt
from .methods.hybrid import SelfCheckHybrid


class ViSelfCheck:
    """
    Main ViSelfCheck class that provides a unified interface to various self-checking methods.
    
    Supported methods:
    - 'bert_score': BERTScore-based self-checking
    - 'nli': Natural Language Inference based self-checking
    - 'mqag': Multiple Choice Question Answering Generation based self-checking
    - 'ngram': N-gram based self-checking
    - 'prompt': API-based prompting self-checking (OpenAI, Groq, Gemini)
    - 'hybrid': Hybrid approach combining NLI and LLM prompting
    """
    
    SUPPORTED_METHODS = {
        'bert_score': SelfCheckBERTScore,
        'nli': SelfCheckNLI,
        'mqag': SelfCheckMQAG,
        'ngram': SelfCheckNgram,
        'prompt': SelfCheckAPIPrompt,
        'hybrid': SelfCheckHybrid
    }
    
    def __init__(self, method: str = 'bert_score', **kwargs):
        """
        Initialize ViSelfCheck with a specific method.
        
        Args:
            method (str): The self-checking method to use. Must be one of:
                         'bert_score', 'nli', 'mqag', 'ngram', 'prompt', 'hybrid'
            **kwargs: Additional keyword arguments specific to the chosen method
        """
        self.method_name = method.lower()
        
        if self.method_name not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. "
                           f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}")
        
        self.method_class = self.SUPPORTED_METHODS[self.method_name]
        self.checker = self._initialize_method(**kwargs)
        
    def _initialize_method(self, **kwargs) -> SelfCheckBase:
        """Initialize the specific method with given parameters."""
        try:
            if self.method_name == 'bert_score':
                return self.method_class(
                    lang=kwargs.get('lang', 'vi'),
                    rescale_with_baseline=kwargs.get('rescale_with_baseline', False),
                    device=kwargs.get('device', None)
                )
            
            elif self.method_name == 'nli':
                return self.method_class(
                    nli_model=kwargs.get('nli_model', None),
                    device=kwargs.get('device', None),
                    do_word_segmentation=kwargs.get('do_word_segmentation', None)
                )
            
            elif self.method_name == 'mqag':
                return self.method_class(
                    QAGenerator_checkpoint=kwargs.get('qa_generator_checkpoint', None),
                    DistractorGenerator_checkpoint=kwargs.get('distractor_generator_checkpoint', None),
                    QuestionAnswerer_checkpoint=kwargs.get('question_answerer_checkpoint', None),
                    QuestionCurator_checkpoint=kwargs.get('question_curator_checkpoint', None),
                    device=kwargs.get('device', None),
                    seed=kwargs.get('seed', 42)
                )
            
            elif self.method_name == 'ngram':
                return self.method_class(
                    n=kwargs.get('n', 1),
                    lowercase=kwargs.get('lowercase', True)
                )
            
            elif self.method_name == 'prompt':
                return self.method_class(
                    client_type=kwargs.get('client_type', 'openai'),
                    model=kwargs.get('model', 'gpt-3.5-turbo'),
                    api_key=kwargs.get('api_key', None)
                )
            
            elif self.method_name == 'hybrid':
                return self.method_class(
                    nli_model=kwargs.get('nli_model', None),
                    device=kwargs.get('device', None),
                    do_word_segmentation=kwargs.get('do_word_segmentation', None),
                    llm_model=kwargs.get('llm_model', None),
                    api_key=kwargs.get('api_key', None)
                )
            
            else:
                raise ValueError(f"Unknown method: {self.method_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.method_name} method: {str(e)}")
    
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        """
        Predict self-consistency scores for given sentences.
        
        Args:
            sentences (List[str]): List of sentences to be evaluated
            sampled_passages (List[str]): List of sampled passages as evidence
            **kwargs: Additional method-specific parameters
            
        Returns:
            List[float]: Sentence-level self-consistency scores
        """
        if not isinstance(sentences, list) or not isinstance(sampled_passages, list):
            raise TypeError("Both sentences and sampled_passages must be lists")
        
        if not sentences:
            raise ValueError("sentences cannot be empty")
        
        if not sampled_passages:
            raise ValueError("sampled_passages cannot be empty")
        
        try:
            return self.checker.predict(sentences, sampled_passages, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Prediction failed with {self.method_name} method: {str(e)}")
    
    def switch_method(self, method: str, **kwargs):
        """
        Switch to a different self-checking method.
        
        Args:
            method (str): The new method to switch to
            **kwargs: Additional keyword arguments for the new method
        """
        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. "
                           f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}")
        
        self.method_name = method.lower()
        self.method_class = self.SUPPORTED_METHODS[self.method_name]
        self.checker = self._initialize_method(**kwargs)
        
        print(f"Switched to {self.method_name} method")
    
    def get_current_method(self) -> str:
        """Get the name of the currently active method."""
        return self.method_name
    
    def get_supported_methods(self) -> List[str]:
        """Get list of all supported methods."""
        return list(self.SUPPORTED_METHODS.keys())
    
    def get_method_info(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific method or current method.
        
        Args:
            method (Optional[str]): Method name to get info for. If None, returns current method info.
            
        Returns:
            Dict[str, Any]: Method information including description and parameters
        """
        target_method = method.lower() if method else self.method_name
        
        if target_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {target_method}")
        
        method_descriptions = {
            'bert_score': {
                'description': 'BERTScore-based self-checking using semantic similarity',
                'parameters': ['lang', 'rescale_with_baseline', 'device'],
                'use_case': 'Good for semantic similarity evaluation'
            },
            'nli': {
                'description': 'Natural Language Inference based self-checking',
                'parameters': ['nli_model', 'device', 'do_word_segmentation'],
                'use_case': 'Good for logical consistency checking'
            },
            'mqag': {
                'description': 'Multiple Choice Question Answering Generation based self-checking',
                'parameters': ['qa_generator_checkpoint', 'distractor_generator_checkpoint', 
                              'question_answerer_checkpoint', 'question_curator_checkpoint', 
                              'device', 'num_questions_per_chunk', 'scoring_method', 'beta1', 'beta2'],
                'use_case': 'Good for factual consistency checking through Q&A'
            },
            'ngram': {
                'description': 'N-gram based self-checking using statistical language models',
                'parameters': ['n', 'lowercase'],
                'use_case': 'Good for lexical consistency checking'
            },
            'prompt': {
                'description': 'API-based prompting self-checking using LLMs',
                'parameters': ['client_type', 'model', 'api_key'],
                'use_case': 'Good for comprehensive evaluation using external LLMs'
            },
            'hybrid': {
                'description': 'Hybrid approach combining NLI and LLM prompting',
                'parameters': ['nli_model', 'device', 'do_word_segmentation', 'llm_model', 'api_key'],
                'use_case': 'Good for combining multiple evaluation approaches'
            }
        }
        
        return method_descriptions[target_method]
    
    def __str__(self) -> str:
        return f"ViSelfCheck(method={self.method_name})"
    
    def __repr__(self) -> str:
        return f"ViSelfCheck(method='{self.method_name}')"


# Convenience functions for quick method creation
def create_bert_score_checker(lang: str = 'vi', rescale_with_baseline: bool = False, device: Optional[str] = None) -> ViSelfCheck:
    """Create a ViSelfCheck instance with BERTScore method."""
    return ViSelfCheck('bert_score', lang=lang, rescale_with_baseline=rescale_with_baseline, device=device)

def create_nli_checker(nli_model: Optional[str] = None, device: Optional[str] = None, do_word_segmentation: Optional[bool] = None) -> ViSelfCheck:
    """Create a ViSelfCheck instance with NLI method."""
    return ViSelfCheck('nli', nli_model=nli_model, device=device, do_word_segmentation=do_word_segmentation)

def create_mqag_checker(device: Optional[str] = None, num_questions_per_chunk: int = 5, 
                       scoring_method: str = 'bayes_with_alpha', **kwargs) -> ViSelfCheck:
    """Create a ViSelfCheck instance with MQAG method."""
    return ViSelfCheck('mqag', device=device, num_questions_per_chunk=num_questions_per_chunk, 
                      scoring_method=scoring_method, **kwargs)

def create_ngram_checker(n: int = 1, lowercase: bool = True) -> ViSelfCheck:
    """Create a ViSelfCheck instance with N-gram method."""
    return ViSelfCheck('ngram', n=n, lowercase=lowercase)

def create_prompt_checker(client_type: str = 'openai', model: str = 'gpt-3.5-turbo', 
                         api_key: Optional[str] = None) -> ViSelfCheck:
    """Create a ViSelfCheck instance with API prompting method."""
    return ViSelfCheck('prompt', client_type=client_type, model=model, api_key=api_key)

def create_hybrid_checker(device: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> ViSelfCheck:
    """Create a ViSelfCheck instance with hybrid method."""
    return ViSelfCheck('hybrid', device=device, api_key=api_key, **kwargs)