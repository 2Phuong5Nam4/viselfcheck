# MQAG Models Comprehensive Evaluation System

## Overview

This comprehensive evaluation system provides a complete solution for evaluating Vietnamese Multiple-choice Question-Answer Generation (MQAG) models. The system evaluates four key components:

1. **QA Generation** - Generates questions and answers from Vietnamese text context
2. **Distractor Generation** - Generates incorrect answer choices for multiple choice questions
3. **MCQ Answering** - Answers multiple choice questions in Vietnamese
4. **Answerability** - Determines if questions can be answered from given context

## Features

- **Vietnamese Language Support**: Specialized for Vietnamese text processing and evaluation
- **Comprehensive Metrics**: Multiple evaluation metrics for each model type
- **Flexible Data Loading**: Support for local files, HuggingFace datasets, and sample data
- **Batch Processing**: Efficient evaluation with configurable batch sizes
- **GPU Support**: Automatic GPU detection with CPU fallback
- **Detailed Logging**: Comprehensive logging and progress tracking
- **Sample Data**: Built-in Vietnamese sample data for testing
- **Multiple Evaluation Modes**: Test, batch, and full evaluation modes

## Requirements

### Essential Dependencies
```bash
pip install torch transformers numpy
```

### Optional Dependencies
```bash
pip install datasets  # For HuggingFace dataset loading
```

## Quick Start

### 1. Basic Test
```bash
# Run a quick test with sample data
python3 mqag_evaluation.py --test
```

### 2. Batch Evaluation
```bash
# Run multiple configurations for testing
python3 mqag_evaluation.py --batch
```

### 3. Full Evaluation
```bash
# With local dataset files
python3 mqag_evaluation.py \
    --viquad-path /path/to/viquad.json \
    --vsmrc-path /path/to/vsmrc.json \
    --batch-size 4 \
    --max-samples 100

# With HuggingFace datasets
python3 mqag_evaluation.py \
    --use-huggingface \
    --max-samples 100 \
    --device cuda
```

## Command Line Arguments

### Basic Arguments
- `--viquad-path PATH`: Path to UIT-ViQuAD dataset JSON file
- `--vsmrc-path PATH`: Path to VSMRC dataset JSON file
- `--batch-size N`: Batch size for evaluation (default: 4)
- `--max-samples N`: Maximum samples to evaluate (default: no limit)
- `--output FILE`: Output JSON file (default: comprehensive_mqag_results.json)
- `--device DEVICE`: Device to use (cuda/cpu, default: auto-detect)
- `--use-huggingface`: Try to load datasets from HuggingFace

### Special Modes
- `--test`: Run quick test evaluation with sample data
- `--batch`: Run batch evaluation with multiple configurations
- `--check-deps`: Check dependencies and exit

## Dataset Formats

### UIT-ViQuAD 2.0 Format
Used for QA Generation and Answerability evaluation:
```json
[
  {
    "context": "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội.",
    "question": "Thủ đô của Việt Nam là gì?",
    "answers": {"text": ["Hà Nội"]},
    "is_impossible": false
  },
  {
    "context": "Phở là món ăn truyền thống của người Việt Nam.",
    "question": "Món ăn gì được coi là truyền thống của Mexico?",
    "answers": {"text": []},
    "is_impossible": true
  }
]
```

### VSMRC Format
Used for Distractor Generation and MCQ Answering evaluation:
```json
[
  {
    "context": "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội.",
    "question": "Thủ đô của Việt Nam là gì?",
    "choices": ["Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Hải Phòng"],
    "correctchoice": 0
  }
]
```

## Model Checkpoints

The system uses these pre-trained Vietnamese models from HuggingFace:

- **QA Generation**: `2Phuong5Nam4/VIT5-base-QA-Generation`
- **Distractor Generation**: `2Phuong5Nam4/VIT5-base-Distractors-Generation`
- **MCQ Answering**: `2Phuong5Nam4/xlm-roberta-base-MCQ-Answering`
- **Answerability**: `2Phuong5Nam4/xlm-roberta-base-Answerable`

## Evaluation Metrics

### QA Generation Metrics
- **Question Exact Match**: Exact string match between generated and reference questions
- **Question F1**: Token-level F1 score for questions
- **Question BLEU**: BLEU score for question generation quality
- **Answer Exact Match**: Exact string match for answers
- **Answer F1**: Token-level F1 score for answers
- **Valid QA Pairs**: Percentage of generated pairs with non-empty questions and answers

### Distractor Generation Metrics
- **Average Exact Matches**: Average number of generated distractors matching reference distractors
- **Max Exact Matches**: Maximum number of matches in any sample
- **Distractor Diversity**: Average ratio of unique distractors in generated sets
- **Semantic Similarity**: Semantic similarity between generated and reference distractors
- **Valid Distractors**: Percentage of generated distractor sets with all non-empty distractors

### MCQ Answering Metrics
- **Accuracy**: Percentage of correctly answered multiple choice questions
- **Average Confidence**: Average confidence score of predictions
- **Confidence Standard Deviation**: Variability in model confidence
- **High Confidence Accuracy**: Accuracy for predictions with confidence > 0.8
- **Choice Accuracies**: Accuracy for each choice position (0-3)

### Answerability Metrics
- **Accuracy**: Percentage of correct answerability predictions
- **Precision**: Precision for answerable questions
- **Recall**: Recall for answerable questions
- **F1**: F1 score for answerability classification
- **Specificity**: Specificity for unanswerable questions
- **Average Score**: Average answerability score
- **Score Standard Deviation**: Variability in answerability scores
- **Answerable Ratio**: Ratio of answerable questions in dataset
- **Predicted Answerable Ratio**: Ratio of questions predicted as answerable

## Sample Output

The evaluation produces a comprehensive JSON report:

```json
{
  "metadata": {
    "device": "cuda:0",
    "batch_size": 4,
    "max_samples": 100,
    "evaluation_time": "2025-07-14 15:30:00"
  },
  "qa_generation": {
    "question_exact_match": 0.1250,
    "question_f1": 0.5432,
    "question_bleu": 0.3456,
    "answer_exact_match": 0.2500,
    "answer_f1": 0.6789,
    "valid_qa_pairs": 0.9500,
    "evaluation_time": 45.67
  },
  "distractor_generation": {
    "avg_exact_matches": 1.2000,
    "max_exact_matches": 3.0000,
    "distractor_diversity": 0.8500,
    "semantic_similarity": 0.4500,
    "valid_distractors": 0.9200,
    "evaluation_time": 32.45
  },
  "mcq_answering": {
    "accuracy": 0.7500,
    "avg_confidence": 0.8234,
    "confidence_std": 0.1567,
    "high_confidence_accuracy": 0.8500,
    "choice_0_accuracy": 0.800,
    "choice_1_accuracy": 0.750,
    "choice_2_accuracy": 0.700,
    "choice_3_accuracy": 0.725,
    "evaluation_time": 28.91
  },
  "answerability": {
    "accuracy": 0.8400,
    "precision": 0.8765,
    "recall": 0.8123,
    "f1": 0.8435,
    "specificity": 0.8750,
    "avg_score": 0.6789,
    "score_std": 0.2345,
    "answerable_ratio": 0.8000,
    "predicted_answerable_ratio": 0.7500,
    "evaluation_time": 21.34
  }
}
```

## Vietnamese Text Processing

The system includes specialized Vietnamese text processing:

### Text Normalization
- Unicode handling for Vietnamese characters
- Punctuation removal and normalization
- Case conversion for consistent comparison
- Whitespace normalization

### Evaluation Metrics
- Token-level comparison for F1 scores
- N-gram based BLEU score calculation
- Semantic similarity assessment
- Vietnamese-specific text matching

## Usage Examples

### Example 1: Quick Test
```bash
# Test the system with built-in sample data
python3 mqag_evaluation.py --test
```

### Example 2: Small Scale Evaluation
```bash
# Evaluate with limited samples for testing
python3 mqag_evaluation.py \
    --use-huggingface \
    --max-samples 50 \
    --batch-size 2 \
    --device cpu
```

### Example 3: Full Production Evaluation
```bash
# Full evaluation with custom datasets
python3 mqag_evaluation.py \
    --viquad-path datasets/uit_viquad_test.json \
    --vsmrc-path datasets/vsmrc_test.json \
    --batch-size 8 \
    --device cuda \
    --output production_results.json
```

### Example 4: Batch Testing
```bash
# Run multiple configurations for comprehensive testing
python3 mqag_evaluation.py --batch
```

## Performance Optimization

### Memory Management
- Configurable batch sizes for memory efficiency
- Model loading on-demand to reduce memory usage
- Efficient data processing for large datasets

### GPU Optimization
- Automatic GPU detection and usage
- Graceful fallback to CPU when GPU unavailable
- Batch processing optimization for GPU acceleration

### Processing Efficiency
- Parallel processing where possible
- Efficient data structures for large datasets
- Progress tracking and logging

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check if you're in the correct directory
   cd /path/to/viselfcheck
   python3 mqag_evaluation.py --check-deps
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python3 mqag_evaluation.py --batch-size 1 --max-samples 10
   ```

3. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python3 mqag_evaluation.py --device cpu
   ```

4. **No Dataset Available**
   ```bash
   # Use sample data for testing
   python3 mqag_evaluation.py --test
   ```

### Performance Tips

1. **Start Small**: Begin with `--max-samples 10` for initial testing
2. **Use GPU**: Enable CUDA for faster evaluation with `--device cuda`
3. **Optimize Batch Size**: Find the optimal batch size for your hardware
4. **Monitor Memory**: Watch GPU/CPU memory usage during evaluation

## Architecture

### Core Components

1. **ComprehensiveMQAGEvaluator**: Main evaluation class
2. **BatchEvaluator**: Batch processing and testing
3. **Utility Functions**: Text processing and metric calculation
4. **CLI Interface**: Command-line argument handling

### Data Flow

1. **Data Loading**: Load from files, HuggingFace, or use sample data
2. **Model Loading**: Load models on-demand for each evaluation
3. **Evaluation**: Run evaluation for each model type
4. **Metrics Calculation**: Calculate comprehensive metrics
5. **Results Storage**: Save results to JSON file
6. **Summary Display**: Print formatted results summary

## Extension and Customization

### Adding New Metrics
```python
def custom_metric(predictions, references):
    # Implement your custom metric
    return score

# Add to evaluation methods
metrics['custom_metric'] = custom_metric(predictions, references)
```

### Adding New Models
```python
def evaluate_new_model(self, data, batch_size=4):
    # Load your model
    # Run evaluation
    # Calculate metrics
    return metrics
```

### Custom Data Processing
```python
def load_custom_data(self, file_path):
    # Implement custom data loading
    # Return data in expected format
    return data
```

## Conclusion

This comprehensive evaluation system provides a robust, production-ready solution for evaluating Vietnamese MQAG models. It combines ease of use with comprehensive evaluation capabilities, making it suitable for both research and production environments.

The system is designed to be:
- **Reliable**: Robust error handling and graceful degradation
- **Scalable**: Efficient processing for large datasets
- **Extensible**: Easy to add new metrics and models
- **User-friendly**: Clear documentation and intuitive interface

For support or questions, please refer to the code comments and logging output for detailed information about the evaluation process.
