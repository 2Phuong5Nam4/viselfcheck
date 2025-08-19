#!/usr/bin/env python3
"""
Test runner for Gemini API-based methods (Prompt and Hybrid).

This script demonstrates how to test the Prompt and Hybrid methods with Gemini API calls.
Set your Gemini API key as an environment variable before running.

Usage:
    export GEMINI_API_KEY="your_gemini_api_key_here"
    python test_gemini_methods.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'viselfcheck', 'src'))

from viselfcheck import ViSelfCheck, create_prompt_checker, create_hybrid_checker


def test_gemini_prompt_method():
    """Test the Prompt method with Gemini API integration."""
    print("🧪 Testing Gemini Prompt Method")
    print("=" * 50)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    try:
        # Initialize prompt checker with Gemini
        checker = ViSelfCheck(
            'prompt',
            model='gemini-2.0-flash',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai',
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
        print("\n🔄 Running Gemini predictions...")
        scores = checker.predict(sentences, passages)
        
        print(f"\n✅ Gemini Prompt method test successful!")
        print(f"📊 Consistency scores: {[f'{score:.4f}' for score in scores]}")
        print(f"📈 Average score: {sum(scores)/len(scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Prompt method test failed: {e}")
        return False


def test_gemini_hybrid_method():
    """Test the Hybrid method with Gemini API integration."""
    print("\n🧪 Testing Gemini Hybrid Method")
    print("=" * 50)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    try:
        # Initialize hybrid checker with Gemini
        checker = ViSelfCheck(
            'hybrid',
            device='cpu',
            nli_model='pgnguyen/phobert-large-nli',
            llm_model='gemini-2.0-flash',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai',
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
        print("\n🔄 Running Gemini predictions...")
        scores = checker.predict(sentences, passages)
        
        print(f"\n✅ Gemini Hybrid method test successful!")
        print(f"📊 Consistency scores: {[f'{score:.4f}' for score in scores]}")
        print(f"📈 Average score: {sum(scores)/len(scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Hybrid method test failed: {e}")
        return False


def test_gemini_convenience_functions():
    """Test convenience functions for Gemini API methods."""
    print("\n🧪 Testing Gemini Convenience Functions")
    print("=" * 50)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    try:
        # Test prompt convenience function with Gemini
        prompt_checker = create_prompt_checker(
            model='gemini-2.0-flash',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai',
            api_key=api_key
        )
        print("✅ Gemini Prompt convenience function works")
        
        # Test hybrid convenience function with Gemini
        hybrid_checker = create_hybrid_checker(
            device='cpu',
            nli_model='pgnguyen/phobert-large-nli',
            llm_model='gemini-2.0-flash',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai',
            do_word_segmentation=True,
            api_key=api_key
        )
        print("✅ Gemini Hybrid convenience function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Convenience function test failed: {e}")
        return False


def test_gemini_env_config():
    """Test Gemini configuration loading from environment variables."""
    print("\n🧪 Testing Gemini Environment Configuration")
    print("=" * 50)
    
    # Check if environment variables are set
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_base_url = os.getenv('GEMINI_BASE_URL')
    gemini_model = os.getenv('GEMINI_MODEL')
    
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        return False
    
    try:
        # Test with environment variables only (specify gemini model to trigger gemini config)
        checker = ViSelfCheck('prompt', model='gemini-2.0-flash')
        print("✅ Gemini configuration loaded from environment variables")
        print(f"   🔑 API Key: {gemini_api_key[:10]}...{gemini_api_key[-5:] if gemini_api_key else 'None'}")
        print(f"   🌐 Base URL: {gemini_base_url or 'Default from env'}")
        print(f"   🤖 Model: {gemini_model or 'Default from env'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini environment configuration test failed: {e}")
        return False


def test_gemini_model_variations():
    """Test different Gemini model variations."""
    print("\n🧪 Testing Gemini Model Variations")
    print("=" * 50)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    gemini_models = [
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    results = []
    
    for model in gemini_models:
        try:
            print(f"\n🔄 Testing model: {model}")
            checker = ViSelfCheck(
                'prompt',
                model=model,
                base_url='https://generativelanguage.googleapis.com/v1beta/openai',
                api_key=api_key
            )
            print(f"✅ Model {model} initialized successfully")
            results.append(True)
            
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            results.append(False)
    
    return all(results)


def main():
    """Run all Gemini API tests."""
    print("🚀 ViSelfCheck Gemini API Methods Test Suite")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No Gemini API key found!")
        print("Please set the environment variable:")
        print("   export GEMINI_API_KEY='your_gemini_api_key_here'")
        return
    
    print(f"🔑 Gemini API key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Run tests
    results = []
    results.append(test_gemini_env_config())
    results.append(test_gemini_model_variations())
    results.append(test_gemini_prompt_method())
    results.append(test_gemini_hybrid_method())
    results.append(test_gemini_convenience_functions())
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    print(f"📈 Success rate: {sum(results)/len(results)*100:.1f}%")
    
    if all(results):
        print("\n🎉 All Gemini API tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
