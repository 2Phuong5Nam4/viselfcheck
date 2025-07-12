"""
ViSelfCheck: A unified interface for various self-checking methods.

This package provides a comprehensive toolkit for self-consistency checking
of generated text using various methods including BERTScore, NLI, MQAG, 
N-gram, API prompting, and hybrid approaches.
"""

from .service import (
    ViSelfCheck,
    create_bert_score_checker,
    create_nli_checker,
    create_mqag_checker,
    create_ngram_checker,
    create_prompt_checker,
    create_hybrid_checker
)

from .methods import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNgram,
    SelfCheckNLI,
    SelfCheckAPIPrompt,
    SelfCheckHybrid
)

from .base import SelfCheckBase

__version__ = "1.0.0"
__author__ = "ViSelfCheck Team"

__all__ = [
    'ViSelfCheck',
    'SelfCheckBase',
    'create_bert_score_checker',
    'create_nli_checker',
    'create_mqag_checker',
    'create_ngram_checker',
    'create_prompt_checker',
    'create_hybrid_checker',
    'SelfCheckBERTScore',
    'SelfCheckMQAG',
    'SelfCheckNgram',
    'SelfCheckNLI',
    'SelfCheckAPIPrompt',
    'SelfCheckHybrid'
]