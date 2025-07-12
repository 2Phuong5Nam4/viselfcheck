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

### Service-Based Usage

The `ViSelfCheck` service provides a unified interface with additional features:

```python
from viselfcheck import ViSelfCheck

# Initialize service with specific method
checker = ViSelfCheck('bert_score', lang='vi', device='cuda')

# Vietnamese text examples
sentences = [
    "Việt Nam là một quốc gia ở Đông Nam Á.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Phở là món ăn truyền thống của Việt Nam."
]

sample_examples = [
    "Việt Nam nằm ở khu vực Đông Nam Á.",
    "Thủ đô của Việt Nam là Hà Nội.",
    "Phở là món ăn nổi tiếng của Việt Nam."
]

# Get consistency scores
scores = checker.predict(sentences, sample_examples)
print(f"Scores: {scores}")

# Service information methods
print(f"Current method: {checker.get_current_method()}")
print(f"Supported methods: {checker.get_supported_methods()}")

# Get method information
info = checker.get_method_info('bert_score')
print(f"Method info: {info}")

# Switch methods dynamically
checker.switch_method('nli', device='cuda')
nli_scores = checker.predict(sentences, sample_examples)
print(f"NLI scores: {nli_scores}")
```

### Factory Functions

Use convenience factory functions for quick setup:

```python
from viselfcheck import (
    create_bert_score_checker,
    create_nli_checker,
    create_mqag_checker,
    create_ngram_checker,
    create_prompt_checker,
    create_hybrid_checker
)

# Create specialized checkers
bert_checker = create_bert_score_checker(lang='vi', device='cuda')
nli_checker = create_nli_checker(device='cuda')
ngram_checker = create_ngram_checker(n=2, lowercase=True)
prompt_checker = create_prompt_checker(client_type='openai', model='gpt-4')

# Use them directly
sentences = ["Việt Nam là quốc gia Đông Nam Á."]
examples = ["Việt Nam nằm ở Đông Nam Á."]

bert_scores = bert_checker.predict(sentences, examples)
nli_scores = nli_checker.predict(sentences, examples)
ngram_scores = ngram_checker.predict(sentences, examples)
prompt_scores = prompt_checker.predict(sentences, examples)

print(f"BERTScore: {bert_scores[0]:.4f}")
print(f"NLI: {nli_scores[0]:.4f}")
print(f"N-gram: {ngram_scores[0]:.4f}")
print(f"Prompt: {prompt_scores[0]:.4f}")
```


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
        
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        """
        Predict consistency scores.
        
        Args:
            sentences: List of sentences to evaluate
            sampled_passages: List of reference examples
            **kwargs: Additional method-specific parameters
            
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
        
    def get_current_method(self) -> str:
        """
        Get the name of the currently active method.
        
        Returns:
            str: Current method name
        """
        
    def get_supported_methods(self) -> List[str]:
        """
        Get list of all supported methods.
        
        Returns:
            List[str]: List of supported method names
        """
        
    def get_method_info(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific method or current method.
        
        Args:
            method: Method name to get info for. If None, returns current method info.
            
        Returns:
            Dict containing description, parameters, and use case
        """
```

### Method-Specific Parameters

#### BERTScore (`bert_score`)
- `lang` (str): Language code (default: 'vi')
- `rescale_with_baseline` (bool): Whether to rescale with baseline (default: False)
- `device` (str|torch.device): Device for computation (default: None)

#### NLI (`nli`)
- `nli_model` (str): NLI model to use (default: None)
- `device` (str|torch.device): Device for computation (default: None)
- `do_word_segmentation` (bool): Whether to perform word segmentation (default: None)

#### MQAG (`mqag`)
- `qa_generator_checkpoint` (str): Path to QA generator checkpoint (default: None)
- `distractor_generator_checkpoint` (str): Path to distractor generator checkpoint (default: None)
- `question_answerer_checkpoint` (str): Path to question answerer checkpoint (default: None)
- `question_curator_checkpoint` (str): Path to question curator checkpoint (default: None)
- `device` (str|torch.device): Device for computation (default: None)
- `seed` (int): Random seed for reproducibility (default: 42)
- `num_questions_per_sent` (int): Number of questions per sentence (default: None)
- `scoring_method` (str): Scoring method - 'counting', 'bayes', or 'bayes_with_alpha' (default: None)
- `AT` (float): Answerability threshold (default: None)
- `beta1` (float): Beta1 parameter for Bayes scoring (default: None)
- `beta2` (float): Beta2 parameter for Bayes scoring (default: None)

#### N-gram (`ngram`)
- `n` (int): N-gram size (default: 1)
- `lowercase` (bool): Whether to convert to lowercase (default: True)

#### Prompt (`prompt`)
- `client_type` (str): API client type - 'openai', 'groq', or 'gemini' (default: 'openai')
- `model` (str): Model name to use (default: 'gpt-3.5-turbo')
- `api_key` (str): API key for authentication (default: None, loads from .env)

#### Hybrid (`hybrid`)
- `nli_model` (str): NLI model to use (default: None)
- `device` (str|torch.device): Device for computation (default: None)
- `do_word_segmentation` (bool): Whether to perform word segmentation (default: None)
- `llm_model` (str): LLM model to use (default: None)
- `api_key` (str): API key for LLM component (default: None)

### Factory Functions

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
checker = create_nli_checker(device='cuda')
checker = create_ngram_checker(n=2, lowercase=True)
checker = create_prompt_checker(client_type='openai', model='gpt-4')
```

---

## 🏆 Real-world Applications

### 1. Content Generation Validation
```python
from viselfcheck import ViSelfCheck

# Initialize service with hybrid method for comprehensive checking
checker = ViSelfCheck('hybrid', device='cuda')

# Validate AI-generated content
generated_articles = get_generated_articles()
reference_sources = get_reference_sources()

scores = checker.predict(generated_articles, reference_sources)

# Filter high-quality content
quality_content = []
for article, score in zip(generated_articles, scores):
    status = "✅ Consistent" if score > 0.8 else "⚠️ Inconsistent"
    print(f"{status} (Score: {score:.3f}): {article[:50]}...")
    
    if score > 0.8:
        quality_content.append(article)

print(f"Accepted {len(quality_content)}/{len(generated_articles)} articles")
```

### 2. Translation Quality Assessment
```python
from viselfcheck import create_bert_score_checker

# Create specialized checker for translation assessment
checker = create_bert_score_checker(lang='vi', device='cuda', rescale_with_baseline=True)

# Assess translation consistency
original_texts = ["Original English text about Vietnam's economy..."]
translated_texts = ["Văn bản tiếng Việt về nền kinh tế Việt Nam..."]

scores = checker.predict(translated_texts, original_texts)

# Evaluate translation quality
for orig, trans, score in zip(original_texts, translated_texts, scores):
    quality = "Excellent" if score > 0.9 else "Good" if score > 0.7 else "Needs Review"
    print(f"Translation Quality: {quality} (Score: {score:.3f})")
    print(f"Original: {orig[:50]}...")
    print(f"Translation: {trans[:50]}...")
```


---

## 🙏 Acknowledgments

- **Underthesea**: Vietnamese NLP library
- **HuggingFace**: Transformers and model hosting
- **OpenAI**: API services
- **Groq**: Fast inference API
- **Google**: Gemini API
