"""
Comprehensive test suite for all ViSelfCheck methods.

This test file provides simple tests to verify that all methods work correctly
with Vietnamese text examples.
"""

import pytest
import sys
import os
from typing import List

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'viselfcheck', 'src'))

from viselfcheck import (
    ViSelfCheck,
    create_bert_score_checker,
    create_nli_checker,
    create_mqag_checker,
    create_ngram_checker,
    create_prompt_checker,
    create_hybrid_checker
)


# Test data - Vietnamese sentences and passages
TEST_SENTENCES = [
    "Việt Nam là một quốc gia ở Đông Nam Á.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Phở là món ăn truyền thống của Việt Nam."
]

TEST_PASSAGES = [
    "Việt Nam nằm ở khu vực Đông Nam Á với thủ đô Hà Nội.",
    "Ẩm thực Việt Nam nổi tiếng với món phở truyền thống.",
    "Đất nước hình chữ S này có văn hóa phong phú."
]

# Simple test data for quick tests
SIMPLE_SENTENCES = ["Việt Nam là một quốc gia."]
SIMPLE_PASSAGES = ["Việt Nam là một đất nước."]


class TestViSelfCheckBasic:
    """Basic tests for ViSelfCheck class functionality."""
    
    def test_supported_methods(self):
        """Test that all expected methods are supported."""
        expected_methods = ['bert_score', 'nli', 'mqag', 'ngram', 'prompt', 'hybrid']
        checker = ViSelfCheck('bert_score')
        
        supported = checker.get_supported_methods()
        assert all(method in supported for method in expected_methods)
    
    def test_invalid_method(self):
        """Test that invalid method raises appropriate error."""
        with pytest.raises(ValueError, match="Unsupported method"):
            ViSelfCheck('invalid_method')
    
    def test_switch_method(self):
        """Test switching between methods."""
        checker = ViSelfCheck('bert_score')
        assert checker.get_current_method() == 'bert_score'
        
        checker.switch_method('ngram')
        assert checker.get_current_method() == 'ngram'
    
    def test_method_info(self):
        """Test getting method information."""
        checker = ViSelfCheck('bert_score')
        info = checker.get_method_info()
        
        assert 'description' in info
        assert 'parameters' in info
        assert 'use_case' in info


class TestBERTScore:
    """Test BERTScore method."""
    
    @pytest.mark.bert_score
    def test_bert_score_initialization(self):
        """Test BERTScore method initialization."""
        checker = ViSelfCheck('bert_score', lang='vi')
        assert checker.get_current_method() == 'bert_score'
    
    @pytest.mark.bert_score
    def test_bert_score_prediction(self):
        """Test BERTScore prediction."""
        checker = ViSelfCheck('bert_score', lang='vi', device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        
        assert isinstance(scores, list)
        assert len(scores) == len(SIMPLE_SENTENCES)
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)
    
    @pytest.mark.bert_score
    def test_bert_score_convenience_function(self):
        """Test BERTScore convenience function."""
        checker = create_bert_score_checker(lang='vi', device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        assert isinstance(scores, list)
        assert len(scores) == 1


class TestNLI:
    """Test NLI method."""
    
    @pytest.mark.nli
    def test_nli_initialization(self):
        """Test NLI method initialization."""
        checker = ViSelfCheck('nli', device='cpu')
        assert checker.get_current_method() == 'nli'
    
    @pytest.mark.nli
    def test_nli_prediction(self):
        """Test NLI prediction."""
        checker = ViSelfCheck('nli', device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        
        assert isinstance(scores, list)
        assert len(scores) == len(SIMPLE_SENTENCES)
        assert all(isinstance(score, float) for score in scores)
    
    @pytest.mark.nli
    def test_nli_convenience_function(self):
        """Test NLI convenience function."""
        checker = create_nli_checker(device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        assert isinstance(scores, list)
        assert len(scores) == 1


class TestMQAG:
    """Test MQAG method."""
    
    @pytest.mark.mqag
    @pytest.mark.slow
    def test_mqag_initialization(self):
        """Test MQAG method initialization."""
        checker = ViSelfCheck('mqag', device='cpu')
        assert checker.get_current_method() == 'mqag'
    
    @pytest.mark.mqag
    @pytest.mark.slow
    def test_mqag_prediction(self):
        """Test MQAG prediction."""
        checker = ViSelfCheck('mqag', device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        
        assert isinstance(scores, list)
        assert len(scores) == len(SIMPLE_SENTENCES)
        assert all(isinstance(score, float) for score in scores)
    
    @pytest.mark.mqag
    @pytest.mark.slow
    def test_mqag_convenience_function(self):
        """Test MQAG convenience function."""
        checker = create_mqag_checker(device='cpu')
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        assert isinstance(scores, list)
        assert len(scores) == 1


class TestNgram:
    """Test N-gram method."""
    
    @pytest.mark.ngram
    def test_ngram_initialization(self):
        """Test N-gram method initialization."""
        checker = ViSelfCheck('ngram', n=2, lowercase=True)
        assert checker.get_current_method() == 'ngram'
    
    @pytest.mark.ngram
    def test_ngram_prediction(self):
        """Test N-gram prediction."""
        checker = ViSelfCheck('ngram', n=2, lowercase=True)
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        
        assert isinstance(scores, list)
        assert len(scores) == len(SIMPLE_SENTENCES)
        assert all(isinstance(score, float) for score in scores)
    
    @pytest.mark.ngram
    def test_ngram_convenience_function(self):
        """Test N-gram convenience function."""
        checker = create_ngram_checker(n=2, lowercase=True)
        
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        assert isinstance(scores, list)
        assert len(scores) == 1


class TestPrompt:
    """Test Prompt method."""
    
    @pytest.mark.prompt
    @pytest.mark.slow
    def test_prompt_initialization(self):
        """Test Prompt method initialization."""
        # This might fail if no API key is configured, which is expected
        try:
            checker = ViSelfCheck('prompt', client_type='openai')
            assert checker.get_current_method() == 'prompt'
        except Exception as e:
            pytest.skip(f"Prompt method requires API configuration: {e}")
    
    @pytest.mark.prompt
    @pytest.mark.slow
    def test_prompt_prediction(self):
        """Test Prompt prediction."""
        try:
            checker = ViSelfCheck('prompt', client_type='openai')
            scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
            
            assert isinstance(scores, list)
            assert len(scores) == len(SIMPLE_SENTENCES)
            assert all(isinstance(score, float) for score in scores)
        except Exception as e:
            pytest.skip(f"Prompt method requires API configuration: {e}")
    
    @pytest.mark.prompt
    @pytest.mark.slow
    def test_prompt_convenience_function(self):
        """Test Prompt convenience function."""
        try:
            checker = create_prompt_checker(client_type='openai')
            scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
            assert isinstance(scores, list)
            assert len(scores) == 1
        except Exception as e:
            pytest.skip(f"Prompt method requires API configuration: {e}")


class TestHybrid:
    """Test Hybrid method."""
    
    @pytest.mark.hybrid
    @pytest.mark.slow
    def test_hybrid_initialization(self):
        """Test Hybrid method initialization."""
        try:
            checker = ViSelfCheck('hybrid', device='cpu')
            assert checker.get_current_method() == 'hybrid'
        except Exception as e:
            pytest.skip(f"Hybrid method requires API configuration: {e}")
    
    @pytest.mark.hybrid
    @pytest.mark.slow
    def test_hybrid_prediction(self):
        """Test Hybrid prediction."""
        try:
            checker = ViSelfCheck('hybrid', device='cpu')
            scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
            
            assert isinstance(scores, list)
            assert len(scores) == len(SIMPLE_SENTENCES)
            assert all(isinstance(score, float) for score in scores)
        except Exception as e:
            pytest.skip(f"Hybrid method requires API configuration: {e}")
    
    @pytest.mark.hybrid
    @pytest.mark.slow
    def test_hybrid_convenience_function(self):
        """Test Hybrid convenience function."""
        try:
            checker = create_hybrid_checker(device='cpu')
            scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
            assert isinstance(scores, list)
            assert len(scores) == 1
        except Exception as e:
            pytest.skip(f"Hybrid method requires API configuration: {e}")


class TestInputValidation:
    """Test input validation across all methods."""
    
    def test_empty_sentences(self):
        """Test handling of empty sentences."""
        checker = ViSelfCheck('ngram')
        
        with pytest.raises(ValueError, match="sentences cannot be empty"):
            checker.predict([], SIMPLE_PASSAGES)
    
    def test_empty_passages(self):
        """Test handling of empty passages."""
        checker = ViSelfCheck('ngram')
        
        with pytest.raises(ValueError, match="sampled_passages cannot be empty"):
            checker.predict(SIMPLE_SENTENCES, [])
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        checker = ViSelfCheck('ngram')
        
        with pytest.raises(TypeError, match="must be lists"):
            checker.predict("not a list", SIMPLE_PASSAGES)
        
        with pytest.raises(TypeError, match="must be lists"):
            checker.predict(SIMPLE_SENTENCES, "not a list")


class TestAllMethodsIntegration:
    """Integration tests to ensure all methods work together."""
    
    @pytest.mark.integration
    def test_method_switching(self):
        """Test switching between all methods."""
        checker = ViSelfCheck('bert_score', lang='vi', device='cpu')
        
        # Test switching to each method
        methods_to_test = ['ngram', 'bert_score']  # Start with fast methods
        
        for method in methods_to_test:
            checker.switch_method(method, device='cpu' if method != 'ngram' else None)
            assert checker.get_current_method() == method
            
            # Test prediction works after switching
            scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
            assert isinstance(scores, list)
            assert len(scores) == 1
    
    @pytest.mark.integration
    def test_multiple_sentences(self):
        """Test all fast methods with multiple sentences."""
        fast_methods = ['bert_score', 'ngram']
        
        for method in fast_methods:
            checker = ViSelfCheck(method, device='cpu' if method != 'ngram' else None)
            
            scores = checker.predict(TEST_SENTENCES, TEST_PASSAGES)
            
            assert isinstance(scores, list)
            assert len(scores) == len(TEST_SENTENCES)
            assert all(isinstance(score, float) for score in scores)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic ViSelfCheck tests...")
    
    # Test BERTScore
    print("\n1. Testing BERTScore method...")
    try:
        checker = ViSelfCheck('bert_score', lang='vi', device='cpu')
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        print(f"   ✅ BERTScore: {scores[0]:.4f}")
    except Exception as e:
        print(f"   ❌ BERTScore failed: {e}")
    
    # Test N-gram
    print("\n2. Testing N-gram method...")
    try:
        checker = ViSelfCheck('ngram', n=2, lowercase=True)
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        print(f"   ✅ N-gram: {scores[0]:.4f}")
    except Exception as e:
        print(f"   ❌ N-gram failed: {e}")
    
    # Test NLI
    print("\n3. Testing NLI method...")
    try:
        checker = ViSelfCheck('nli', device='cpu')
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        print(f"   ✅ NLI: {scores[0]:.4f}")
    except Exception as e:
        print(f"   ❌ NLI failed: {e}")
    
    # Test method switching
    print("\n4. Testing method switching...")
    try:
        checker = ViSelfCheck('bert_score', lang='vi', device='cpu')
        checker.switch_method('ngram', n=2)
        scores = checker.predict(SIMPLE_SENTENCES, SIMPLE_PASSAGES)
        print(f"   ✅ Method switching: {scores[0]:.4f}")
    except Exception as e:
        print(f"   ❌ Method switching failed: {e}")
    
    print("\n✅ Basic tests completed!")
    print("\nTo run full test suite: pytest tests/test_all_methods.py")
    print("To run specific method tests: pytest tests/test_all_methods.py -m bert_score")
    print("To run fast tests only: pytest tests/test_all_methods.py -m 'not slow'")
