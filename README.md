# ViSelfCheck

A unified interface for various self-checking methods to evaluate text consistency in Vietnamese text.

## 🎯 Overview

ViSelfCheck provides a comprehensive toolkit for self-consistency checking of generated text using various methods including:

- **BERTScore**: Semantic similarity-based checking
- **NLI**: Natural Language Inference based checking  
- **MQAG**: Multiple Choice Question Answering Generation based checking
- **N-gram**: Statistical language model based checking
- **Prompt**: API-based prompting using external LLMs (OpenAI, Groq, Gemini)
- **Hybrid**: Combined approach using multiple methods

### Key Features
- **6 Self-Checking Methods**: BERTScore, NLI, MQAG, N-gram, Prompt, Hybrid
- **Vietnamese Support**: Optimized for Vietnamese text processing
- **Unified Interface**: Switch between methods dynamically
- **Production Ready**: Comprehensive error handling and type safety

---

## ⚠️ Python 3.8 Required

**This package requires Python 3.8 specifically** due to the `underthesea` library dependency for Vietnamese text processing.

### Why Python 3.8 Only?

The `underthesea` library is essential for Vietnamese text processing and is used in multiple methods:
- **BERTScore**: Vietnamese sentence tokenization
- **NLI**: Vietnamese word segmentation  
- **Hybrid**: Vietnamese text preprocessing
- **N-gram**: Vietnamese text processing

Unfortunately, `underthesea` only supports Python 3.8 and is not compatible with Python 3.9+ or Python 3.7-.

---

## 🚀 Installation

### Prerequisites

Before installing, ensure you have the required system dependencies:

```bash
# Ubuntu/Debian - Install system dependencies
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.8 if needed
# Ubuntu/Debian
sudo apt install python3.8
sudo apt install python3.8-dev python3.8-venv python3.8-distutils


# Windows
# Download from https://www.python.org/downloads/release/python-3815/
```

### Quick Install

```bash
# 1. Check Python version
python3.8 --version  # Must show Python 3.8.x

# 2. Create virtual environment
python3.8 -m venv venv_viselfcheck

# 3. Activate virtual environment
source venv_viselfcheck/bin/activate  # Linux/macOS
# venv_viselfcheck\Scripts\activate   # Windows

# 4. Install package
cd /path/to/viselfcheck  # Navigate to the package directory where pyproject.toml is located
pip install -e .


# 5. Run all tests with pytest
cd tests  # Navigate to the tests directory
python -m pytest
```

### API Configuration (for Prompt & Hybrid methods)

To use API-based methods, configure your API keys:

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Example `.env` configuration:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-openai-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Groq API Configuration  
GROQ_API_KEY=gsk_your-actual-groq-key-here
GROQ_MODEL=llama3-8b-8192

# Google Gemini API Configuration
GEMINI_API_KEY=your-actual-gemini-key-here
GEMINI_MODEL=gemini-pro
```

```bash
# 3. Test API configuration
python test_api.py
```

That's it! Everything will be installed automatically.

---

## 💡 Usage Examples

### Basic Usage

```python
from viselfcheck import ViSelfCheck

# Initialize with BERTScore method
checker = ViSelfCheck('bert_score', lang='vi')

# Vietnamese text example
sentences = ["Việt Nam là một quốc gia ở Đông Nam Á."]
sample_examples = ["Việt Nam nằm ở khu vực Đông Nam Á."]

# Get consistency scores
scores = checker.predict(sentences, sample_examples)
print(f"Consistency scores: {scores}")

# Switch to different method
checker.switch_method('nli')
scores = checker.predict(sentences, sample_examples)
print(f"NLI scores: {scores}")
```

### Advanced Usage

```python
from viselfcheck import ViSelfCheck

# Initialize with specific configurations
checker = ViSelfCheck(
    method='bert_score',
    lang='vi',
    rescale_with_baseline=True,
    device='cuda'  # Use GPU if available
)

# Multiple sentences evaluation
sentences = [
    "Việt Nam là một quốc gia ở Đông Nam Á.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Phở là món ăn truyền thống của Việt Nam."
]

sample_examples = [
    "Việt Nam nằm ở khu vực Đông Nam Á với thủ đô Hà Nội.",
    "Ẩm thực Việt Nam nổi tiếng với món phở truyền thống.",
    "Đất nước hình chữ S này có văn hóa phong phú."
]

# Get detailed results
scores = checker.predict(sentences, sample_examples)

# Process results
for i, (sentence, score) in enumerate(zip(sentences, scores)):
    status = "✅ Consistent" if score > 0.7 else "⚠️ Inconsistent"
    print(f"{i+1}. {status} (Score: {score:.3f})")
    print(f"   {sentence}")
```

### API-based Methods

```python
from viselfcheck import ViSelfCheck

# Using OpenAI API (auto-loads from .env)
checker = ViSelfCheck('prompt')  # Uses .env configuration
scores = checker.predict(sentences, sample_examples)

# Or specify API type explicitly
checker = ViSelfCheck('prompt', client_type='openai')
scores = checker.predict(sentences, sample_examples)

# Using Groq API
checker = ViSelfCheck('prompt', client_type='groq')
scores = checker.predict(sentences, sample_examples)

# Using Google Gemini API
checker = ViSelfCheck('prompt', client_type='gemini')
scores = checker.predict(sentences, sample_examples)

# Test different APIs and compare
apis = ['openai', 'groq', 'gemini']
for api in apis:
    try:
        checker = ViSelfCheck('prompt', client_type=api)
        scores = checker.predict(sentences, sample_examples)
        print(f"{api}: {scores[0]:.4f}")
    except Exception as e:
        print(f"{api}: Not configured")
```

---

## 🔧 Available Methods

### 1. BERTScore (`bert_score`)
- **Description**: Semantic similarity-based checking using BERT models
- **Best for**: Semantic consistency evaluation
- **Parameters**: `lang`, `rescale_with_baseline`, `device`

### 2. NLI (`nli`)
- **Description**: Natural Language Inference based checking
- **Best for**: Logical consistency evaluation
- **Parameters**: `device`

### 3. MQAG (`mqag`)
- **Description**: Multiple Choice Question Answering Generation
- **Best for**: Fact-based consistency evaluation
- **Parameters**: `device`

### 4. N-gram (`ngram`)
- **Description**: Statistical language model based checking
- **Best for**: Language model consistency evaluation
- **Parameters**: `n` (gram size), `lowercase`

### 5. Prompt (`prompt`)
- **Description**: API-based prompting using external LLMs
- **Best for**: Advanced consistency evaluation with latest models
- **Parameters**: `api_type`, `api_key`, `model`

### 6. Hybrid (`hybrid`)
- **Description**: Combined approach using multiple methods
- **Best for**: Comprehensive consistency evaluation
- **Parameters**: Combination of above methods

---

## � Performance Comparison

| Method | Speed | Accuracy | Vietnamese Support | GPU Required |
|--------|-------|----------|-------------------|--------------|
| BERTScore | Fast | Medium | ✅ | Recommended |
| NLI | Fast | High | ✅ | Recommended |
| MQAG | Slow | Medium | ✅ | Recommended |
| N-gram | Very Fast | Medium | ✅ | No |
| Prompt | Medium | Very High | ✅ | No |
| Hybrid | Medium | Very High | ✅ | Recommended |

---



## 🚨 Troubleshooting

### Common Issues

#### 1. Python Version Error
```bash
# Error: Package requires Python 3.8
# Solution: Install Python 3.8
sudo apt install python3.8 python3.8-venv
```

#### 2. PyTorch Installation Issues
```bash
# Error: No module named '_ctypes'
# Solution: Install system dependencies
sudo apt install libffi-dev python3.8-dev
```

#### 3. Underthesea Import Error
```bash
# Error: No module named '_sqlite3'
# Solution: Install sqlite3 development headers
sudo apt install libsqlite3-dev
```

#### 4. BERT Score Import Error
```bash
# Error: No module named '_bz2'
# Solution: Install bzip2 development headers
sudo apt install libbz2-dev
```

### Performance Issues

#### GPU Not Detected
```python
# Check GPU availability
import torch
print("CUDA available:", torch.cuda.is_available())

# Force CPU usage if needed
checker = ViSelfCheck('bert_score', device='cpu')
```

#### Memory Issues
```python
# Process in smaller batches
def process_large_dataset(sentences, sample_examples, batch_size=32):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_examples = sample_examples[i:i+batch_size]
        scores = checker.predict(batch_sentences, batch_examples)
        results.extend(scores)
    return results
```

---

## 📝 API Reference

### ViSelfCheck Class

```python
class ViSelfCheck:
    def __init__(self, method: str, **kwargs):
        """
        Initialize ViSelfCheck with specified method.
        
        Args:
            method: One of 'bert_score', 'nli', 'mqag', 'ngram', 'prompt', 'hybrid'
            **kwargs: Method-specific parameters
        """
        
    def predict(self, sentences: List[str], sample_examples: List[str]) -> List[float]:
        """
        Predict consistency scores.
        
        Args:
            sentences: List of sentences to evaluate
            sample_examples: List of reference examples
            
        Returns:
            List of consistency scores (0-1, higher = more consistent)
        """
        
    def switch_method(self, method: str, **kwargs):
        """
        Switch to a different method.
        
        Args:
            method: New method name
            **kwargs: Method-specific parameters
        """
```

### Helper Functions

```python
# Create method-specific checkers
from viselfcheck import (
    create_bert_score_checker,
    create_nli_checker,
    create_mqag_checker,
    create_ngram_checker,
    create_prompt_checker,
    create_hybrid_checker
)

# Example usage
checker = create_bert_score_checker(lang='vi', device='cuda')
```

---

## 🏆 Real-world Applications

### 1. Content Generation Validation
```python
# Validate AI-generated content
generated_articles = get_generated_articles()
reference_sources = get_reference_sources()

checker = ViSelfCheck('hybrid')
scores = checker.predict(generated_articles, reference_sources)

# Filter high-quality content
quality_content = [
    article for article, score in zip(generated_articles, scores)
    if score > 0.8
]
```

### 2. Translation Quality Assessment
```python
# Assess translation consistency
original_texts = ["Original English text..."]
translated_texts = ["Bản dịch tiếng Việt..."]

checker = ViSelfCheck('bert_score', lang='vi')
scores = checker.predict(translated_texts, original_texts)
```

### 3. Chatbot Response Validation
```python
# Validate chatbot responses
user_queries = ["Việt Nam có bao nhiêu tỉnh thành?"]
bot_responses = ["Việt Nam có 63 tỉnh thành."]
knowledge_base = ["Việt Nam gồm 63 tỉnh thành phố..."]

checker = ViSelfCheck('nli')
scores = checker.predict(bot_responses, knowledge_base)
```

---

## 🙏 Acknowledgments

- **Underthesea**: Vietnamese NLP library
- **HuggingFace**: Transformers and model hosting
- **OpenAI**: API services
- **Groq**: Fast inference API
- **Google**: Gemini API
