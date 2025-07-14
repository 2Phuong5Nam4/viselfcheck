#!/usr/bin/env python3
"""
MQAG Models Comprehensive Evaluation Script
===========================================

This script provides a complete evaluation system for four Vietnamese MQAG models:
1. QA Generation - Generates questions and answers from context
2. Distractor Generation - Generates incorrect answer choices
3. MCQ Answering - Answers multiple choice questions
4. Answerability - Determines if questions can be answered from context

Author: GitHub Copilot
Date: July 14, 2025
Usage: python3 mqag_evaluation.py [options]
"""

import torch
import json
import argparse
import subprocess
import time
from pathlib import Path
import sys
import os
import numpy as np
from collections import Counter
import string
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "viselfcheck" / "src"))

try:
    from viselfcheck.modeling.modeling_mqag import (
        QAGenerator, 
        DistractorGenerator, 
        QuestionAnswerer, 
        QuestionCurator
    )
except ImportError:
    print("Error: Cannot import modeling modules. Please check the path.")
    print("Make sure you're running this script from the correct directory.")
    sys.exit(1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match score"""
    matches = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_text(pred)
        ref_norm = normalize_text(ref)
        matches.append(1 if pred_norm == ref_norm else 0)
    return float(np.mean(matches))


def compute_f1_score(prediction: str, reference: str) -> float:
    """Compute F1 score between prediction and reference"""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    common = pred_counter & ref_counter
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_batch_f1(predictions: List[str], references: List[str]) -> float:
    """Compute average F1 score for a batch"""
    f1_scores = []
    for pred, ref in zip(predictions, references):
        f1_scores.append(compute_f1_score(pred, ref))
    return float(np.mean(f1_scores))


def compute_bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
    """Simple BLEU score implementation"""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return Counter(ngrams)
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            scores.append(0.0)
            continue
        
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams or not ref_ngrams:
            scores.append(0.0)
            continue
        
        # Calculate precision
        common = pred_ngrams & ref_ngrams
        precision = sum(common.values()) / sum(pred_ngrams.values())
        
        # Simple brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens))
        
        scores.append(bp * precision)
    
    return float(np.mean(scores))


def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['torch', 'transformers', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return False
    
    return True


# ============================================================================
# MAIN EVALUATOR CLASS
# ============================================================================

class ComprehensiveMQAGEvaluator:
    """Comprehensive evaluator for all MQAG models"""
    
    def __init__(self, device: Optional[torch.device] = None, max_samples: Optional[int] = None):
        """
        Initialize the evaluator
        
        Args:
            device: torch device to use
            max_samples: maximum number of samples to evaluate
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_samples = max_samples
        
        # Model checkpoints
        self.checkpoints = {
            'qa_generation': "2Phuong5Nam4/VIT5-base-QA-Generation",
            'distractor_generation': "2Phuong5Nam4/VIT5-base-Distractors-Generation", 
            'mcq_answering': "2Phuong5Nam4/xlm-roberta-base-MCQ-Answering",
            'answerability': "2Phuong5Nam4/xlm-roberta-base-Answerable"
        }
        
        # Initialize models as None
        self.qa_generator = None
        self.distractor_generator = None
        self.question_answerer = None
        self.question_curator = None
        
        logger.info(f"Initialized evaluator with device: {self.device}")
        if self.max_samples:
            logger.info(f"Limited to {self.max_samples} samples")
    
    def load_qa_generator(self):
        """Load QA Generator model"""
        if self.qa_generator is None:
            logger.info("Loading QA Generator...")
            self.qa_generator = QAGenerator(
                checkpoint_path=self.checkpoints['qa_generation'],
                device=self.device
            )
    
    def load_distractor_generator(self):
        """Load Distractor Generator model"""
        if self.distractor_generator is None:
            logger.info("Loading Distractor Generator...")
            self.distractor_generator = DistractorGenerator(
                checkpoint_path=self.checkpoints['distractor_generation'],
                device=self.device
            )
    
    def load_question_answerer(self):
        """Load Question Answerer model"""
        if self.question_answerer is None:
            logger.info("Loading Question Answerer...")
            self.question_answerer = QuestionAnswerer(
                checkpoint_path=self.checkpoints['mcq_answering'],
                device=self.device
            )
    
    def load_question_curator(self):
        """Load Question Curator model"""
        if self.question_curator is None:
            logger.info("Loading Question Curator...")
            self.question_curator = QuestionCurator(
                checkpoint_path=self.checkpoints['answerability'],
                device=self.device
            )
    
    def create_sample_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Create sample Vietnamese data for testing"""
        logger.info("Creating sample Vietnamese data for testing...")
        
        # Sample ViQuAD-style data
        viquad_sample = [
            {
                'context': 'Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội. Hà Nội là thành phố lớn thứ hai của Việt Nam sau Thành phố Hồ Chí Minh.',
                'question': 'Thủ đô của Việt Nam là gì?',
                'answers': {'text': ['Hà Nội']},
                'is_impossible': False
            },
            {
                'context': 'Sông Mekong là con sông dài nhất ở Đông Nam Á với chiều dài khoảng 4.350 km. Sông này chảy qua 6 quốc gia bao gồm Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia và Việt Nam.',
                'question': 'Sông nào dài nhất ở Đông Nam Á?',
                'answers': {'text': ['Sông Mekong']},
                'is_impossible': False
            },
            {
                'context': 'Phở là món ăn truyền thống của người Việt Nam. Phở bò và phở gà là hai loại phở phổ biến nhất. Món ăn này được làm từ bánh phở, nước dầm và thịt.',
                'question': 'Món ăn gì được coi là truyền thống của Mexico?',
                'answers': {'text': []},
                'is_impossible': True
            },
            {
                'context': 'Trường Đại học Bách khoa Hà Nội được thành lập năm 1956. Đây là một trong những trường đại học kỹ thuật hàng đầu của Việt Nam.',
                'question': 'Trường Đại học Bách khoa Hà Nội được thành lập năm nào?',
                'answers': {'text': ['1956']},
                'is_impossible': False
            },
            {
                'context': 'Cà phê Việt Nam nổi tiếng trên thế giới. Việt Nam là nước xuất khẩu cà phê lớn thứ hai thế giới sau Brazil.',
                'question': 'Việt Nam là nước xuất khẩu cà phê thứ mấy trên thế giới?',
                'answers': {'text': ['thứ hai']},
                'is_impossible': False
            }
        ]
        
        # Sample VSMRC-style data
        vsmrc_sample = [
            {
                'context': 'Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội. Hà Nội là thành phố lớn thứ hai của Việt Nam.',
                'question': 'Thủ đô của Việt Nam là gì?',
                'choices': ['Hà Nội', 'Thành phố Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng'],
                'correctchoice': 0
            },
            {
                'context': 'Sông Mekong là con sông dài nhất ở Đông Nam Á với chiều dài khoảng 4.350 km. Sông này chảy qua 6 quốc gia.',
                'question': 'Sông nào dài nhất ở Đông Nam Á?',
                'choices': ['Sông Hồng', 'Sông Mekong', 'Sông Đồng Nai', 'Sông Cửu Long'],
                'correctchoice': 1
            },
            {
                'context': 'Trường Đại học Bách khoa Hà Nội được thành lập năm 1956. Đây là một trong những trường đại học kỹ thuật hàng đầu.',
                'question': 'Trường Đại học Bách khoa Hà Nội được thành lập năm nào?',
                'choices': ['1954', '1956', '1960', '1945'],
                'correctchoice': 1
            },
            {
                'context': 'Cà phê Việt Nam nổi tiếng trên thế giới. Việt Nam là nước xuất khẩu cà phê lớn thứ hai thế giới sau Brazil.',
                'question': 'Việt Nam là nước xuất khẩu cà phê thứ mấy trên thế giới?',
                'choices': ['thứ nhất', 'thứ hai', 'thứ ba', 'thứ tư'],
                'correctchoice': 1
            }
        ]
        
        if self.max_samples:
            viquad_sample = viquad_sample[:self.max_samples]
            vsmrc_sample = vsmrc_sample[:self.max_samples]
        
        return viquad_sample, vsmrc_sample
    
    def load_data_from_file(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.max_samples:
                data = data[:self.max_samples]
            
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def load_data_from_huggingface(self, dataset_name: str, split: str = "test") -> List[Dict]:
        """Load data from HuggingFace datasets"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            data = [dict(sample) for sample in dataset]
            
            if self.max_samples:
                data = data[:self.max_samples]
            
            logger.info(f"Loaded {len(data)} samples from HuggingFace dataset: {dataset_name}")
            return data
        except ImportError:
            logger.warning("datasets library not available. Cannot load from HuggingFace.")
            return []
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name}: {e}")
            return []
    
    def evaluate_qa_generation(self, viquad_data: List[Dict], batch_size: int = 4) -> Dict[str, float]:
        """Evaluate QA Generation model"""
        logger.info("Evaluating QA Generation...")
        self.load_qa_generator()
        
        contexts = [item['context'] for item in viquad_data]
        reference_questions = [item['question'] for item in viquad_data]
        reference_answers = [
            item['answers']['text'][0] if item['answers']['text'] else "" 
            for item in viquad_data
        ]
        
        # Generate questions and answers
        logger.info(f"Generating questions and answers for {len(contexts)} samples...")
        generated_questions, generated_answers = self.qa_generator.generate(
            contexts, batch_size=batch_size
        )
        
        # Calculate metrics
        metrics = {
            'question_exact_match': compute_exact_match(generated_questions, reference_questions),
            'question_f1': compute_batch_f1(generated_questions, reference_questions),
            'question_bleu': compute_bleu_score(generated_questions, reference_questions),
            'answer_exact_match': compute_exact_match(generated_answers, reference_answers),
            'answer_f1': compute_batch_f1(generated_answers, reference_answers),
            'valid_qa_pairs': sum(1 for q, a in zip(generated_questions, generated_answers) if q.strip() and a.strip()) / len(generated_questions)
        }
        
        # Log some examples
        logger.info("Sample generated QA pairs:")
        for i in range(min(3, len(generated_questions))):
            logger.info(f"Q{i+1}: {generated_questions[i]}")
            logger.info(f"A{i+1}: {generated_answers[i]}")
        
        logger.info("QA Generation evaluation completed")
        return metrics
    
    def evaluate_distractor_generation(self, vsmrc_data: List[Dict], batch_size: int = 4) -> Dict[str, float]:
        """Evaluate Distractor Generation model"""
        logger.info("Evaluating Distractor Generation...")
        self.load_distractor_generator()
        
        contexts = [item['context'] for item in vsmrc_data]
        questions = [item['question'] for item in vsmrc_data]
        answers = [item['choices'][item['correctchoice']] for item in vsmrc_data]
        
        # Get true distractors
        true_distractors = []
        for item in vsmrc_data:
            choices = item['choices']
            correct = choices[item['correctchoice']]
            true_distractors.append([choice for choice in choices if choice != correct])
        
        # Generate distractors
        logger.info(f"Generating distractors for {len(contexts)} samples...")
        generated_distractors = self.distractor_generator.generate(
            contexts, questions, answers, batch_size=batch_size
        )
        
        # Calculate metrics
        match_counts = []
        diversity_scores = []
        semantic_similarity_scores = []
        
        for pred_dist, true_dist in zip(generated_distractors, true_distractors):
            # Exact matches
            exact_matches = len(set(pred_dist) & set(true_dist))
            match_counts.append(exact_matches)
            
            # Diversity within generated distractors
            unique_distractors = len(set(pred_dist))
            total_distractors = len(pred_dist)
            diversity_scores.append(unique_distractors / total_distractors if total_distractors > 0 else 0)
            
            # Semantic similarity (simple token-based)
            semantic_sim = 0
            for pred in pred_dist:
                for true in true_dist:
                    if compute_f1_score(pred, true) > 0.5:  # threshold for similarity
                        semantic_sim += 1
                        break
            semantic_similarity_scores.append(semantic_sim / len(pred_dist) if pred_dist else 0)
        
        metrics = {
            'avg_exact_matches': float(np.mean(match_counts)),
            'max_exact_matches': float(np.max(match_counts)) if match_counts else 0,
            'distractor_diversity': float(np.mean(diversity_scores)),
            'semantic_similarity': float(np.mean(semantic_similarity_scores)),
            'valid_distractors': sum(1 for dist_list in generated_distractors if all(d.strip() for d in dist_list)) / len(generated_distractors)
        }
        
        # Log some examples
        logger.info("Sample generated distractors:")
        for i in range(min(3, len(generated_distractors))):
            logger.info(f"Sample {i+1}: {generated_distractors[i]}")
        
        logger.info("Distractor Generation evaluation completed")
        return metrics
    
    def evaluate_mcq_answering(self, vsmrc_data: List[Dict], batch_size: int = 4) -> Dict[str, float]:
        """Evaluate MCQ Answering model"""
        logger.info("Evaluating MCQ Answering...")
        self.load_question_answerer()
        
        contexts = [item['context'] for item in vsmrc_data]
        questions = [item['question'] for item in vsmrc_data]
        options = [item['choices'] for item in vsmrc_data]
        true_indices = [item['correctchoice'] for item in vsmrc_data]
        
        # Predict answers
        logger.info(f"Predicting MCQ answers for {len(contexts)} samples...")
        probabilities = self.question_answerer.predict(
            contexts, questions, options, batch_size=batch_size
        )
        
        # Get predicted indices
        predicted_indices = [int(torch.argmax(torch.tensor(prob))) for prob in probabilities]
        
        # Calculate metrics
        correct_predictions = [
            1 if pred == true else 0 
            for pred, true in zip(predicted_indices, true_indices)
        ]
        
        max_confidences = [max(prob) for prob in probabilities]
        
        # Calculate per-choice accuracy
        choice_accuracies = [0, 0, 0, 0]  # for 4 choices
        choice_counts = [0, 0, 0, 0]
        
        for true_idx, pred_idx in zip(true_indices, predicted_indices):
            if true_idx < 4:  # ensure valid index
                choice_counts[true_idx] += 1
                if pred_idx == true_idx:
                    choice_accuracies[true_idx] += 1
        
        # Normalize choice accuracies
        for i in range(4):
            if choice_counts[i] > 0:
                choice_accuracies[i] /= choice_counts[i]
        
        metrics = {
            'accuracy': float(np.mean(correct_predictions)),
            'avg_confidence': float(np.mean(max_confidences)),
            'confidence_std': float(np.std(max_confidences)),
            'high_confidence_accuracy': float(np.mean([correct for correct, conf in zip(correct_predictions, max_confidences) if conf > 0.8])),
            'choice_0_accuracy': choice_accuracies[0],
            'choice_1_accuracy': choice_accuracies[1],
            'choice_2_accuracy': choice_accuracies[2],
            'choice_3_accuracy': choice_accuracies[3]
        }
        
        # Log some examples
        logger.info("Sample MCQ predictions:")
        for i in range(min(3, len(predicted_indices))):
            logger.info(f"Sample {i+1}: Predicted={predicted_indices[i]}, True={true_indices[i]}, Confidence={max_confidences[i]:.3f}")
        
        logger.info("MCQ Answering evaluation completed")
        return metrics
    
    def evaluate_answerability(self, viquad_data: List[Dict], batch_size: int = 4, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate Answerability model"""
        logger.info("Evaluating Answerability...")
        self.load_question_curator()
        
        contexts = [item['context'] for item in viquad_data]
        questions = [item['question'] for item in viquad_data]
        true_labels = [0 if item.get('is_impossible', False) else 1 for item in viquad_data]
        
        # Predict answerability scores
        logger.info(f"Predicting answerability scores for {len(contexts)} samples...")
        scores = self.question_curator.predict(
            contexts, questions, batch_size=batch_size
        )
        
        # Convert to binary predictions
        predicted_labels = [1 if score > threshold else 0 for score in scores]
        
        # Calculate metrics
        correct_predictions = [
            1 if pred == true else 0 
            for pred, true in zip(predicted_labels, true_labels)
        ]
        
        # Calculate precision, recall, F1
        true_positives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == 1 and true == 1)
        false_positives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == 1 and true == 0)
        false_negatives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == 0 and true == 1)
        true_negatives = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == 0 and true == 0)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        metrics = {
            'accuracy': float(np.mean(correct_predictions)),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'avg_score': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'answerable_ratio': float(np.mean(true_labels)),
            'predicted_answerable_ratio': float(np.mean(predicted_labels))
        }
        
        # Log some examples
        logger.info("Sample answerability predictions:")
        for i in range(min(3, len(predicted_labels))):
            logger.info(f"Sample {i+1}: Predicted={predicted_labels[i]}, True={true_labels[i]}, Score={scores[i]:.3f}")
        
        logger.info("Answerability evaluation completed")
        return metrics
    
    def run_comprehensive_evaluation(self, viquad_path: Optional[str] = None, vsmrc_path: Optional[str] = None, 
                                   batch_size: int = 4, output_file: str = "comprehensive_mqag_results.json",
                                   use_huggingface: bool = False) -> Dict:
        """Run comprehensive evaluation of all models"""
        logger.info("Starting comprehensive MQAG evaluation...")
        
        results = {
            'metadata': {
                'device': str(self.device),
                'batch_size': batch_size,
                'max_samples': self.max_samples,
                'checkpoints': self.checkpoints,
                'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Load data
        viquad_data = []
        vsmrc_data = []
        
        if use_huggingface:
            logger.info("Attempting to load data from HuggingFace...")
            viquad_data = self.load_data_from_huggingface("taidng/UIT-ViQuAD2.0", "test")
            vsmrc_data = self.load_data_from_huggingface("VSMRC/mrc", "test")
        
        if not viquad_data and viquad_path:
            viquad_data = self.load_data_from_file(viquad_path)
        
        if not vsmrc_data and vsmrc_path:
            vsmrc_data = self.load_data_from_file(vsmrc_path)
        
        # Use sample data if no data loaded
        if not viquad_data or not vsmrc_data:
            logger.info("Using sample data for evaluation...")
            sample_viquad, sample_vsmrc = self.create_sample_data()
            if not viquad_data:
                viquad_data = sample_viquad
            if not vsmrc_data:
                vsmrc_data = sample_vsmrc
        
        if not viquad_data or not vsmrc_data:
            logger.error("No data available for evaluation")
            return {}
        
        logger.info(f"Using {len(viquad_data)} ViQuAD samples and {len(vsmrc_data)} VSMRC samples")
        
        try:
            # 1. QA Generation
            logger.info("=" * 60)
            start_time = time.time()
            qa_metrics = self.evaluate_qa_generation(viquad_data, batch_size)
            results['qa_generation'] = qa_metrics
            results['qa_generation']['evaluation_time'] = time.time() - start_time
            
            # 2. Distractor Generation
            logger.info("=" * 60)
            start_time = time.time()
            dist_metrics = self.evaluate_distractor_generation(vsmrc_data, batch_size)
            results['distractor_generation'] = dist_metrics
            results['distractor_generation']['evaluation_time'] = time.time() - start_time
            
            # 3. MCQ Answering
            logger.info("=" * 60)
            start_time = time.time()
            mcq_metrics = self.evaluate_mcq_answering(vsmrc_data, batch_size)
            results['mcq_answering'] = mcq_metrics
            results['mcq_answering']['evaluation_time'] = time.time() - start_time
            
            # 4. Answerability
            logger.info("=" * 60)
            start_time = time.time()
            answerability_metrics = self.evaluate_answerability(viquad_data, batch_size)
            results['answerability'] = answerability_metrics
            results['answerability']['evaluation_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
        
        # Save results
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        self.print_comprehensive_summary(results)
        
        return results
    
    def print_comprehensive_summary(self, results: Dict):
        """Print comprehensive evaluation results summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MQAG MODELS EVALUATION SUMMARY")
        print("=" * 80)
        
        # Metadata
        if 'metadata' in results:
            meta = results['metadata']
            print(f"\nEvaluation Details:")
            print(f"  Device: {meta.get('device', 'unknown')}")
            print(f"  Batch Size: {meta.get('batch_size', 'unknown')}")
            print(f"  Max Samples: {meta.get('max_samples', 'no limit')}")
            print(f"  Evaluation Time: {meta.get('evaluation_time', 'unknown')}")
        
        # QA Generation
        if 'qa_generation' in results:
            print("\n📝 QA GENERATION RESULTS:")
            qa = results['qa_generation']
            print(f"  Question Exact Match: {qa.get('question_exact_match', 0):.4f}")
            print(f"  Question F1: {qa.get('question_f1', 0):.4f}")
            print(f"  Question BLEU: {qa.get('question_bleu', 0):.4f}")
            print(f"  Answer Exact Match: {qa.get('answer_exact_match', 0):.4f}")
            print(f"  Answer F1: {qa.get('answer_f1', 0):.4f}")
            print(f"  Valid QA Pairs: {qa.get('valid_qa_pairs', 0):.4f}")
            print(f"  Evaluation Time: {qa.get('evaluation_time', 0):.2f}s")
        
        # Distractor Generation
        if 'distractor_generation' in results:
            print("\n🎯 DISTRACTOR GENERATION RESULTS:")
            dist = results['distractor_generation']
            print(f"  Avg Exact Matches: {dist.get('avg_exact_matches', 0):.4f}")
            print(f"  Max Exact Matches: {dist.get('max_exact_matches', 0):.4f}")
            print(f"  Distractor Diversity: {dist.get('distractor_diversity', 0):.4f}")
            print(f"  Semantic Similarity: {dist.get('semantic_similarity', 0):.4f}")
            print(f"  Valid Distractors: {dist.get('valid_distractors', 0):.4f}")
            print(f"  Evaluation Time: {dist.get('evaluation_time', 0):.2f}s")
        
        # MCQ Answering
        if 'mcq_answering' in results:
            print("\n❓ MCQ ANSWERING RESULTS:")
            mcq = results['mcq_answering']
            print(f"  Accuracy: {mcq.get('accuracy', 0):.4f}")
            print(f"  Avg Confidence: {mcq.get('avg_confidence', 0):.4f}")
            print(f"  Confidence Std: {mcq.get('confidence_std', 0):.4f}")
            print(f"  High Confidence Accuracy: {mcq.get('high_confidence_accuracy', 0):.4f}")
            print(f"  Choice Accuracies: [{mcq.get('choice_0_accuracy', 0):.3f}, {mcq.get('choice_1_accuracy', 0):.3f}, {mcq.get('choice_2_accuracy', 0):.3f}, {mcq.get('choice_3_accuracy', 0):.3f}]")
            print(f"  Evaluation Time: {mcq.get('evaluation_time', 0):.2f}s")
        
        # Answerability
        if 'answerability' in results:
            print("\n✅ ANSWERABILITY RESULTS:")
            ans = results['answerability']
            print(f"  Accuracy: {ans.get('accuracy', 0):.4f}")
            print(f"  Precision: {ans.get('precision', 0):.4f}")
            print(f"  Recall: {ans.get('recall', 0):.4f}")
            print(f"  F1: {ans.get('f1', 0):.4f}")
            print(f"  Specificity: {ans.get('specificity', 0):.4f}")
            print(f"  Avg Score: {ans.get('avg_score', 0):.4f}")
            print(f"  Score Std: {ans.get('score_std', 0):.4f}")
            print(f"  Answerable Ratio: {ans.get('answerable_ratio', 0):.4f}")
            print(f"  Predicted Answerable Ratio: {ans.get('predicted_answerable_ratio', 0):.4f}")
            print(f"  Evaluation Time: {ans.get('evaluation_time', 0):.2f}s")
        
        print("\n" + "=" * 80)
        print("Evaluation completed successfully! 🎉")
        print("=" * 80)


# ============================================================================
# BATCH EVALUATION AND TESTING
# ============================================================================

class BatchEvaluator:
    """Batch evaluator for running multiple configurations"""
    
    def __init__(self):
        self.evaluator = None
    
    def run_test_evaluation(self, device: str = "cpu", max_samples: int = 2) -> bool:
        """Run a quick test evaluation"""
        logger.info(f"Running test evaluation with device={device}, max_samples={max_samples}")
        
        try:
            self.evaluator = ComprehensiveMQAGEvaluator(
                device=torch.device(device),
                max_samples=max_samples
            )
            
            results = self.evaluator.run_comprehensive_evaluation(
                batch_size=1,
                output_file=f"test_results_{device}_{max_samples}.json"
            )
            
            if results:
                logger.info("✓ Test evaluation completed successfully")
                return True
            else:
                logger.error("✗ Test evaluation failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Test evaluation failed with error: {e}")
            return False
    
    def run_batch_evaluation(self) -> bool:
        """Run batch evaluation with multiple configurations"""
        logger.info("Starting batch evaluation...")
        
        configs = [
            {'name': 'Quick Test (CPU, 2 samples)', 'device': 'cpu', 'max_samples': 2, 'batch_size': 1},
            {'name': 'Small Test (CPU, 5 samples)', 'device': 'cpu', 'max_samples': 5, 'batch_size': 2},
        ]
        
        # Add GPU config if available
        if torch.cuda.is_available():
            configs.append({
                'name': 'GPU Test (CUDA, 5 samples)', 
                'device': 'cuda', 
                'max_samples': 5, 
                'batch_size': 4
            })
        
        success_count = 0
        total_count = len(configs)
        
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Running configuration {i+1}/{total_count}: {config['name']}")
            print(f"{'='*60}")
            
            try:
                self.evaluator = ComprehensiveMQAGEvaluator(
                    device=torch.device(config['device']),
                    max_samples=config['max_samples']
                )
                
                start_time = time.time()
                results = self.evaluator.run_comprehensive_evaluation(
                    batch_size=config['batch_size'],
                    output_file=f"batch_results_{i+1}.json"
                )
                end_time = time.time()
                
                if results:
                    success_count += 1
                    logger.info(f"✓ Configuration completed in {end_time - start_time:.2f}s")
                else:
                    logger.error(f"✗ Configuration failed")
                    
            except Exception as e:
                logger.error(f"✗ Configuration failed with error: {e}")
        
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {success_count}/{total_count}")
        print(f"Failed: {total_count - success_count}/{total_count}")
        
        if success_count == total_count:
            print("🎉 All configurations completed successfully!")
            return True
        else:
            print("❌ Some configurations failed!")
            return False


# ============================================================================
# MAIN FUNCTION AND CLI
# ============================================================================

def main():
    """Main function with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Comprehensive MQAG Models Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sample data
  python3 mqag_evaluation.py --test
  
  # Batch evaluation
  python3 mqag_evaluation.py --batch
  
  # Full evaluation with custom data
  python3 mqag_evaluation.py --viquad-path data/viquad.json --vsmrc-path data/vsmrc.json
  
  # Evaluation with HuggingFace datasets
  python3 mqag_evaluation.py --use-huggingface --max-samples 100
        """
    )
    
    # Main arguments
    parser.add_argument("--viquad-path", type=str, help="Path to UIT-ViQuAD dataset JSON file")
    parser.add_argument("--vsmrc-path", type=str, help="Path to VSMRC dataset JSON file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation (default: 4)")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate (default: no limit)")
    parser.add_argument("--output", type=str, default="comprehensive_mqag_results.json", help="Output JSON file")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--use-huggingface", action="store_true", help="Try to load datasets from HuggingFace")
    
    # Special modes
    parser.add_argument("--test", action="store_true", help="Run quick test evaluation")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation with multiple configurations")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        if check_dependencies():
            print("✓ All dependencies are available")
            return 0
        else:
            return 1
    
    # Check basic dependencies
    if not check_dependencies():
        return 1
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle special modes
    if args.test:
        batch_evaluator = BatchEvaluator()
        success = batch_evaluator.run_test_evaluation(device=str(device).split(':')[0])
        return 0 if success else 1
    
    if args.batch:
        batch_evaluator = BatchEvaluator()
        success = batch_evaluator.run_batch_evaluation()
        return 0 if success else 1
    
    # Regular evaluation
    try:
        evaluator = ComprehensiveMQAGEvaluator(device=device, max_samples=args.max_samples)
        
        results = evaluator.run_comprehensive_evaluation(
            viquad_path=args.viquad_path,
            vsmrc_path=args.vsmrc_path,
            batch_size=args.batch_size,
            output_file=args.output,
            use_huggingface=args.use_huggingface
        )
        
        if results:
            logger.info("Evaluation completed successfully!")
            return 0
        else:
            logger.error("Evaluation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
