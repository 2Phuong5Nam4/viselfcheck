#!/usr/bin/env python3
"""
Test runner for OpenAI API-based methods (Prompt and Hybrid).

This script demonstrates how to test the Prompt and Hybrid methods with OpenAI API calls.
Set your OpenAI API key as an environment variable before running.

Usage:
    export OPENAI_API_KEY="your_openai_api_key_here"
    python test_openai_methods.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'viselfcheck', 'src'))

from viselfcheck import ViSelfCheck, create_prompt_checker, create_hybrid_checker


def test_openai_prompt_method():
    """Test the Prompt method with OpenAI API integration."""
    print("🧪 Testing OpenAI Prompt Method")
    print("=" * 50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    try:
        # Initialize prompt checker with OpenAI
        checker = ViSelfCheck(
            'prompt',
            model='gpt-3.5-turbo',
            base_url='https://api.openai.com/v1',
            api_key=api_key
        )
        
        # Test data
        sentences = [
            "Việt Nam là một quốc gia ở Đông Nam Á.",
            "Hà Nội là thủ đô của Việt Nam."
        ]
        
        passages = [
            "Việt Nam nằm ở khu vực Đông Nam Á với thủ đô Hà Nội.",
            "Thủ đô Hà Nội của Việt Nam là trung tâm chính trị và văn hóa."
        ]
        
        print("📝 Test sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"   {i}. {sentence}")
        
        print("\n📄 Test passages:")
        for i, passage in enumerate(passages, 1):
            print(f"   {i}. {passage}")
        
        # Run prediction
        print("\n🔄 Running OpenAI predictions...")
        scores = checker.predict(sentences, passages)
        
        print(f"\n✅ OpenAI Prompt method test successful!")
        print(f"📊 Consistency scores: {[f'{score:.4f}' for score in scores]}")
        print(f"📈 Average score: {sum(scores)/len(scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI Prompt method test failed: {e}")
        return False


def test_openai_hybrid_method():
    """Test the Hybrid method with OpenAI API integration."""
    print("\n🧪 Testing OpenAI Hybrid Method")
    print("=" * 50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    try:
        # Initialize hybrid checker with OpenAI
        checker = ViSelfCheck(
            'hybrid',
            device='cpu',
            nli_model='pgnguyen/phobert-large-nli',
            llm_model='gpt-3.5-turbo',
            base_url='https://api.openai.com/v1',
            do_word_segmentation=True,
            api_key=api_key
        )
        
        # Test data
        sentences = [
            "Phở là món ăn truyền thống của Việt Nam.",
            "Sài Gòn là thành phố lớn nhất Việt Nam."
        ]
        
        passages = [
            "Ẩm thực Việt Nam nổi tiếng với món phở truyền thống và bánh mì.",
            "Thành phố Hồ Chí Minh (Sài Gòn) là trung tâm kinh tế lớn nhất Việt Nam."
        ]
        
        print("📝 Test sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"   {i}. {sentence}")
        
        print("\n📄 Test passages:")
        for i, passage in enumerate(passages, 1):
            print(f"   {i}. {passage}")
        
        # Run prediction
        print("\n🔄 Running OpenAI predictions...")
        scores = checker.predict(sentences, passages)
        
        print(f"\n✅ OpenAI Hybrid method test successful!")
        print(f"📊 Consistency scores: {[f'{score:.4f}' for score in scores]}")
        print(f"📈 Average score: {sum(scores)/len(scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI Hybrid method test failed: {e}")
        return False


def test_openai_convenience_functions():
    """Test convenience functions for OpenAI API methods."""
    print("\n🧪 Testing OpenAI Convenience Functions")
    print("=" * 50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    try:
        # Test prompt convenience function with OpenAI
        prompt_checker = create_prompt_checker(
            model='gpt-3.5-turbo',
            base_url='https://api.openai.com/v1',
            api_key=api_key
        )
        print("✅ OpenAI Prompt convenience function works")
        
        # Test hybrid convenience function with OpenAI
        hybrid_checker = create_hybrid_checker(
            device='cpu',
            nli_model='pgnguyen/phobert-large-nli',
            llm_model='gpt-3.5-turbo',
            base_url='https://api.openai.com/v1',
            do_word_segmentation=True,
            api_key=api_key
        )
        print("✅ OpenAI Hybrid convenience function works")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI Convenience function test failed: {e}")
        return False


def test_openai_env_config():
    """Test OpenAI configuration loading from environment variables."""
    print("\n🧪 Testing OpenAI Environment Configuration")
    print("=" * 50)
    
    # Check if environment variables are set
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_base_url = os.getenv('OPENAI_BASE_URL')
    openai_model = os.getenv('OPENAI_MODEL')
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables.")
        return False
    
    try:
        # Test with environment variables only (no explicit parameters)
        checker = ViSelfCheck('prompt')
        print("✅ OpenAI configuration loaded from environment variables")
        print(f"   🔑 API Key: {openai_api_key[:10]}...{openai_api_key[-5:] if openai_api_key else 'None'}")
        print(f"   🌐 Base URL: {openai_base_url or 'Default from env'}")
        print(f"   🤖 Model: {openai_model or 'Default from env'}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI environment configuration test failed: {e}")
        return False


def main():
    """Run all OpenAI API tests."""
    print("🚀 ViSelfCheck OpenAI API Methods Test Suite")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Please set the environment variable:")
        print("   export OPENAI_API_KEY='your_openai_api_key_here'")
        return
    
    print(f"🔑 OpenAI API key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Run tests
    results = []
    results.append(test_openai_env_config())
    results.append(test_openai_prompt_method())
    results.append(test_openai_hybrid_method())
    results.append(test_openai_convenience_functions())
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    print(f"📈 Success rate: {sum(results)/len(results)*100:.1f}%")
    
    if all(results):
        print("\n🎉 All OpenAI API tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
