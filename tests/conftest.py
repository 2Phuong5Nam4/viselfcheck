import pytest

# Register all custom markers used in the test files
def pytest_configure(config):
    """Register custom markers for the test suite."""
    # Method-specific markers
    config.addinivalue_line("markers", "bert_score: tests for BERTScore method")
    config.addinivalue_line("markers", "nli: tests for NLI method")
    config.addinivalue_line("markers", "mqag: tests for MQAG method")
    config.addinivalue_line("markers", "ngram: tests for N-gram method")
    config.addinivalue_line("markers", "prompt: tests for Prompt method")
    config.addinivalue_line("markers", "hybrid: tests for Hybrid method")
    
    # Test type markers
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped in quick mode)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")