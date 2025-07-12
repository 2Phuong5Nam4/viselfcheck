from typing import List, Optional, Union, Dict, Any
import torch

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

        Raises:
            ValueError: If the specified method is not supported
        """
        self.method_name = method.lower()

        if self.method_name not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}"
            )

        self.method_class = self.SUPPORTED_METHODS[self.method_name]
        self.checker = self._initialize_method(**kwargs)

    # ================================
    # Private Helper Methods
    # ================================

    def _handle_device(self, device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
        """
        Convert device string to torch.device object if needed.
        
        Args:
            device: Device specification as string or torch.device object
            
        Returns:
            torch.device object or None
            
        Raises:
            ValueError: If device type is invalid
        """
        if device is None:
            return None
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, torch.device):
            return device
        else:
            raise ValueError(
                f"Invalid device type: {type(device)}. Expected str or torch.device."
            )

    def _initialize_method(self, **kwargs) -> SelfCheckBase:
        """
        Initialize the specific method with given parameters.
        
        Args:
            **kwargs: Method-specific parameters
            
        Returns:
            SelfCheckBase: Initialized method instance
            
        Raises:
            RuntimeError: If method initialization fails
        """
        try:
            if self.method_name == 'bert_score':
                return self.method_class(
                    lang=kwargs.get('lang', 'vi'),
                    rescale_with_baseline=kwargs.get('rescale_with_baseline', False),
                    device=self._handle_device(kwargs.get('device', None))
                )

            elif self.method_name == 'nli':
                return self.method_class(
                    nli_model=kwargs.get('nli_model', None),
                    device=self._handle_device(kwargs.get('device', None)),
                    do_word_segmentation=kwargs.get('do_word_segmentation', None)
                )

            elif self.method_name == 'mqag':
                return self.method_class(
                    QAGenerator_checkpoint=kwargs.get('qa_generator_checkpoint', None),
                    DistractorGenerator_checkpoint=kwargs.get('distractor_generator_checkpoint', None),
                    QuestionAnswerer_checkpoint=kwargs.get('question_answerer_checkpoint', None),
                    QuestionCurator_checkpoint=kwargs.get('question_curator_checkpoint', None),
                    device=self._handle_device(kwargs.get('device', None)),
                    seed=kwargs.get('seed', 42),
                    num_questions_per_sent=kwargs.get('num_questions_per_sent', None),
                    scoring_method=kwargs.get('scoring_method', None),
                    AT=kwargs.get('AT', None),
                    beta1=kwargs.get('beta1', None),
                    beta2=kwargs.get('beta2', None)
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
                    device=self._handle_device(kwargs.get('device', None)),
                    do_word_segmentation=kwargs.get('do_word_segmentation', None),
                    llm_model=kwargs.get('llm_model', None),
                    api_key=kwargs.get('api_key', None)
                )

            else:
                raise ValueError(f"Unknown method: {self.method_name}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {self.method_name} method: {str(e)}"
            )

    # ================================
    # Public Methods
    # ================================

    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        """
        Predict self-consistency scores for given sentences.

        Args:
            sentences (List[str]): List of sentences to be evaluated
            sampled_passages (List[str]): List of sampled passages as evidence
            **kwargs: Additional method-specific parameters

        Returns:
            List[float]: Sentence-level self-consistency scores
            
        Raises:
            TypeError: If input arguments are not lists
            ValueError: If input lists are empty
            RuntimeError: If prediction fails
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
            raise RuntimeError(
                f"Prediction failed with {self.method_name} method: {str(e)}"
            )

    def switch_method(self, method: str, **kwargs):
        """
        Switch to a different self-checking method.

        Args:
            method (str): The new method to switch to
            **kwargs: Additional keyword arguments for the new method
            
        Raises:
            ValueError: If the specified method is not supported
        """
        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}"
            )

        self.method_name = method.lower()
        self.method_class = self.SUPPORTED_METHODS[self.method_name]
        self.checker = self._initialize_method(**kwargs)

        print(f"Switched to {self.method_name} method")

    # ================================
    # Information Methods
    # ================================

    def get_current_method(self) -> str:
        """
        Get the name of the currently active method.
        
        Returns:
            str: Current method name
        """
        return self.method_name

    def get_supported_methods(self) -> List[str]:
        """
        Get list of all supported methods.
        
        Returns:
            List[str]: List of supported method names
        """
        return list(self.SUPPORTED_METHODS.keys())

    def get_method_info(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific method or current method.

        Args:
            method (Optional[str]): Method name to get info for. 
                                   If None, returns current method info.

        Returns:
            Dict[str, Any]: Method information including description and parameters
            
        Raises:
            ValueError: If the specified method is unknown
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
                'parameters': [
                    'qa_generator_checkpoint', 'distractor_generator_checkpoint',
                    'question_answerer_checkpoint', 'question_curator_checkpoint',
                    'device', 'seed', 'num_questions_per_sent', 'scoring_method', 'AT', 'beta1', 'beta2'
                ],
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

    # ================================
    # String Representations
    # ================================

    def __str__(self) -> str:
        return f"ViSelfCheck(method={self.method_name})"

    def __repr__(self) -> str:
        return f"ViSelfCheck(method='{self.method_name}')"


# ================================
# Convenience Factory Functions
# ================================

def create_bert_score_checker(
    lang: str = 'vi',
    rescale_with_baseline: bool = False,
    device: Optional[Union[str, torch.device]] = None
) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with BERTScore method.
    
    Args:
        lang: Language code
        rescale_with_baseline: Whether to rescale with baseline
        device: Device to use for computation
        
    Returns:
        ViSelfCheck: Configured BERTScore checker
    """
    return ViSelfCheck(
        'bert_score',
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        device=device
    )


def create_nli_checker(
    nli_model: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    do_word_segmentation: Optional[bool] = None
) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with NLI method.
    
    Args:
        nli_model: NLI model to use
        device: Device to use for computation
        do_word_segmentation: Whether to perform word segmentation
        
    Returns:
        ViSelfCheck: Configured NLI checker
    """
    return ViSelfCheck(
        'nli',
        nli_model=nli_model,
        device=device,
        do_word_segmentation=do_word_segmentation
    )


def create_mqag_checker(
    qa_generator_checkpoint: Optional[str] = None,
    distractor_generator_checkpoint: Optional[str] = None,
    question_answerer_checkpoint: Optional[str] = None,
    question_curator_checkpoint: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    seed: int = 42,
    num_questions_per_sent: Optional[int] = None,
    scoring_method: Optional[str] = None,
    AT: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None
) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with MQAG method.
    
    Args:
        qa_generator_checkpoint: Path to QA generator checkpoint
        distractor_generator_checkpoint: Path to distractor generator checkpoint
        question_answerer_checkpoint: Path to question answerer checkpoint
        question_curator_checkpoint: Path to question curator checkpoint
        device: Device to use for computation
        seed: Random seed for reproducibility
        num_questions_per_sent: Number of questions to be generated per sentence
        scoring_method: Scoring method - 'counting', 'bayes', or 'bayes_with_alpha'
        AT: Answerability threshold
        beta1: Beta1 parameter for Bayes scoring
        beta2: Beta2 parameter for Bayes scoring
        
    Returns:
        ViSelfCheck: Configured MQAG checker
    """
    return ViSelfCheck(
        'mqag',
        qa_generator_checkpoint=qa_generator_checkpoint,
        distractor_generator_checkpoint=distractor_generator_checkpoint,
        question_answerer_checkpoint=question_answerer_checkpoint,
        question_curator_checkpoint=question_curator_checkpoint,
        device=device,
        seed=seed,
        num_questions_per_sent=num_questions_per_sent,
        scoring_method=scoring_method,
        AT=AT,
        beta1=beta1,
        beta2=beta2
    )


def create_ngram_checker(n: int = 1, lowercase: bool = True) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with N-gram method.
    
    Args:
        n: N-gram size
        lowercase: Whether to convert to lowercase
        
    Returns:
        ViSelfCheck: Configured N-gram checker
    """
    return ViSelfCheck('ngram', n=n, lowercase=lowercase)


def create_prompt_checker(
    client_type: str = 'openai',
    model: str = 'gpt-3.5-turbo',
    api_key: Optional[str] = None
) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with API prompting method.
    
    Args:
        client_type: Type of API client
        model: Model to use
        api_key: API key for authentication
        
    Returns:
        ViSelfCheck: Configured prompt checker
    """
    return ViSelfCheck(
        'prompt',
        client_type=client_type,
        model=model,
        api_key=api_key
    )


def create_hybrid_checker(
    nli_model: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    do_word_segmentation: Optional[bool] = None,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None
) -> ViSelfCheck:
    """
    Create a ViSelfCheck instance with hybrid method.
    
    Args:
        nli_model: NLI model to use
        device: Device to use for computation
        do_word_segmentation: Whether to perform word segmentation
        llm_model: LLM model to use
        api_key: API key for LLM component
        
    Returns:
        ViSelfCheck: Configured hybrid checker
    """
    return ViSelfCheck(
        'hybrid',
        nli_model=nli_model,
        device=device,
        do_word_segmentation=do_word_segmentation,
        llm_model=llm_model,
        api_key=api_key
    )
