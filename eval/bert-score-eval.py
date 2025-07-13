#!/usr/bin/env python3
"""
BERTScore evaluation script for multiple languages.

This script evaluates BERTScore with different language models and configurations
to test performance across multiple languages using the Vietnamese Hallucination dataset.
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from typing import Dict, List, Any, Optional

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'viselfcheck', 'src'))

from viselfcheck import ViSelfCheck
from utils import read_dataset, result_collect


def clone_dataset(dataset_path: str = "Vietnamese_Hallucination"):
    """Clone the Vietnamese Hallucination dataset if it doesn't exist."""
    if not os.path.exists(dataset_path):
        print(f"Cloning dataset from HuggingFace...")
        os.system(f"git clone https://huggingface.co/datasets/2Phuong5Nam4/{dataset_path}")
    else:
        print(f"Dataset {dataset_path} already exists.")


def evaluate_bert_score_model(
    model_name: str, 
    dataset, 
    device, 
    rescale_with_baseline: bool = True,
    hash_file: Optional[str] = None,
    language_override: Optional[str] = None
):
    """
    Evaluate BERTScore with a specific model.
    
    Args:
        model_name: Name/path of the BERT model to use
        dataset: Dataset to evaluate on
        device: Device to use for computation
        rescale_with_baseline: Whether to rescale with baseline
        hash_file: Path to hash file for baseline rescaling
        language_override: Override automatic language detection
        
    Returns:
        Dictionary with BERTScore results
    """
    print("=" * 60)
    print(f"Evaluating BERTScore with model: {model_name}")
    print(f"Rescale with baseline: {rescale_with_baseline}")
    if language_override:
        print(f"Language override: {language_override}")
    
    # Initialize BERTScore checker with specific model
    bert_score_params = {
        'rescale_with_baseline': rescale_with_baseline,
        'model_type': model_name,
        'device': device
    }
    
    if hash_file:
        bert_score_params['hash_file'] = hash_file
    
    if language_override:
        bert_score_params['lang'] = language_override
    
    try:
        checker = ViSelfCheck(method='bert_score', **bert_score_params)
        scores = {}
        
        num_datapoints = len(dataset)
        with tqdm(total=num_datapoints, desc=f'BERTScore-{model_name}') as pbar:
            for i, datapoint in enumerate(dataset):
                try:
                    selfcheck_scores_ = checker.predict(
                        sentences=datapoint['gemini_sentences'],
                        sampled_passages=datapoint['gemini_text_samples']
                    )
                    
                    # Handle None or invalid scores
                    if selfcheck_scores_ is None:
                        print(f"Warning: Got None result for datapoint {i+1}")
                        selfcheck_scores_ = [0.0] * len(datapoint['gemini_sentences'])
                    elif any(score is None for score in selfcheck_scores_):
                        print(f"Warning: Got None scores in result for datapoint {i+1}")
                        selfcheck_scores_ = [0.0 if score is None else score for score in selfcheck_scores_]
                    
                    scores[i] = selfcheck_scores_
                    
                except Exception as e:
                    print(f"Error processing datapoint {i+1}: {e}")
                    scores[i] = [0.0] * len(datapoint['gemini_sentences'])
                
                pbar.update(1)
        
        return scores
        
    except Exception as e:
        print(f"Error initializing BERTScore with model {model_name}: {e}")
        return None


def evaluate_all_bert_score_models(dataset, device, model_configs):
    """
    Evaluate BERTScore with multiple models and configurations.
    
    Args:
        dataset: Dataset to evaluate on
        device: Device to use for computation
        model_configs: Dictionary of model configurations
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for config_name, config in model_configs.items():
        try:
            print(f"\n{'='*70}")
            print(f"EVALUATING BERT-SCORE CONFIG: {config_name.upper()}")
            print(f"{'='*70}")
            
            # Extract model parameters
            model_name = config['model_name']
            params = config.get('params', {})
            result_name = config.get('result_name', f"BERTScore-{model_name}")
            
            # Evaluate the model
            scores = evaluate_bert_score_model(
                model_name=model_name,
                dataset=dataset,
                device=device,
                **params
            )
            
            if scores is not None:
                # Collect results
                result = result_collect(scores, dataset, result_name)
                all_results.append(result)
                
                print(f"Results for {result_name}:")
                for key, value in result.items():
                    if key != 'Method':
                        print(f"  {key}: {value:.2f}")
            else:
                print(f"Failed to evaluate {config_name}")
                
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def get_language_model_configs():
    """
    Define configurations for different language models to test.
    
    Returns:
        Dictionary of model configurations
    """
    configs = {
        # Multilingual models
        'mbert_base': {
            'model_name': 'bert-base-multilingual-cased',
            'params': {'rescale_with_baseline': True},
            'result_name': 'mBERT-base'
        },
        'mbert_base_no_rescale': {
            'model_name': 'bert-base-multilingual-cased',
            'params': {'rescale_with_baseline': False},
            'result_name': 'mBERT-base-NoRescale'
        },
        'xlm_roberta_base': {
            'model_name': 'xlm-roberta-base',
            'params': {'rescale_with_baseline': True},
            'result_name': 'XLM-R-base'
        },
        'xlm_roberta_large': {
            'model_name': 'xlm-roberta-large',
            'params': {'rescale_with_baseline': True},
            'result_name': 'XLM-R-large'
        },
        
        # Vietnamese-specific models
        'phobert_base': {
            'model_name': 'vinai/phobert-base',
            'params': {'rescale_with_baseline': True, 'language_override': 'vi'},
            'result_name': 'PhoBERT-base'
        },
        'phobert_large': {
            'model_name': 'vinai/phobert-large',
            'params': {'rescale_with_baseline': True, 'language_override': 'vi'},
            'result_name': 'PhoBERT-large'
        },
        'bartpho_base': {
            'model_name': 'vinai/bartpho-syllable-base',
            'params': {'rescale_with_baseline': True, 'language_override': 'vi'},
            'result_name': 'BARTPho-base'
        },
        
        # English models for comparison
        'bert_base_uncased': {
            'model_name': 'bert-base-uncased',
            'params': {'rescale_with_baseline': True, 'language_override': 'en'},
            'result_name': 'BERT-base-uncased'
        },
        'roberta_base': {
            'model_name': 'roberta-base',
            'params': {'rescale_with_baseline': True, 'language_override': 'en'},
            'result_name': 'RoBERTa-base'
        },
        'roberta_large': {
            'model_name': 'roberta-large',
            'params': {'rescale_with_baseline': True, 'language_override': 'en'},
            'result_name': 'RoBERTa-large'
        },
        
        # Additional multilingual models
        'distilbert_multilingual': {
            'model_name': 'distilbert-base-multilingual-cased',
            'params': {'rescale_with_baseline': True},
            'result_name': 'DistilBERT-multilingual'
        }
    }
    
    return configs


def test_quick_models(dataset, device, num_samples: int = 5):
    """Test a few models quickly with a small dataset sample."""
    print("=" * 60)
    print("QUICK TEST WITH SELECTED MODELS")
    print("=" * 60)
    
    # Use only first few datapoints for quick testing
    test_size = min(num_samples, len(dataset))
    test_dataset = [dataset[i] for i in range(test_size)]
    
    # Convert to Dataset object
    from datasets import Dataset
    test_dataset_obj = Dataset.from_list(test_dataset)
    
    # Test configurations (subset of models for quick testing)
    quick_configs = {
        'mbert_base': {
            'model_name': 'bert-base-multilingual-cased',
            'params': {'rescale_with_baseline': True},
            'result_name': 'mBERT-base'
        },
        'xlm_roberta_base': {
            'model_name': 'xlm-roberta-base',
            'params': {'rescale_with_baseline': True},
            'result_name': 'XLM-R-base'
        },
        'phobert_base': {
            'model_name': 'vinai/phobert-base',
            'params': {'rescale_with_baseline': True, 'language_override': 'vi'},
            'result_name': 'PhoBERT-base'
        }
    }
    
    results = evaluate_all_bert_score_models(test_dataset_obj, device, quick_configs)
    
    if results:
        df_result = pd.DataFrame(results).set_index('Method').round(2)
        print("\n" + "=" * 60)
        print("QUICK TEST RESULTS")
        print("=" * 60)
        print(df_result)
        return df_result
    
    return None


def main():
    """Main function to run BERTScore evaluations with multiple languages."""
    parser = argparse.ArgumentParser(description='Evaluate BERTScore with multiple language models')
    parser.add_argument('--dataset-path', default='Vietnamese_Hallucination', 
                       help='Path to the dataset directory')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with subset of models and data')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples for quick test (default: 5)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--models', nargs='+', 
                       default=['mbert_base', 'xlm_roberta_base', 'phobert_base', 'roberta_base'],
                       help='Model configurations to evaluate')
    parser.add_argument('--output-file', default='bert_score_results.csv',
                       help='Output CSV file name')
    parser.add_argument('--include-baseline-comparison', action='store_true',
                       help='Include both rescaled and non-rescaled versions')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Clone and load dataset
    clone_dataset(args.dataset_path)
    print(f"Loading dataset from {args.dataset_path}...")
    
    # Find the CSV file in the dataset directory
    csv_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.dataset_path}")
    
    dataset_csv_path = os.path.join(args.dataset_path, csv_files[0])
    print(f"Found CSV file: {dataset_csv_path}")
    
    dataset = read_dataset(dataset_csv_path)
    print(f"Dataset loaded with {len(dataset)} datapoints")
    
    # Quick test mode
    if args.quick_test:
        result_df = test_quick_models(dataset, device, args.num_samples)
        if result_df is not None:
            test_output_file = f'bert_score_quick_test_results.csv'
            result_df.to_csv(test_output_file)
            print(f"\nQuick test results saved to {test_output_file}")
        return
    
    # Get all model configurations
    all_model_configs = get_language_model_configs()
    
    # Filter models based on user selection
    filtered_configs = {k: v for k, v in all_model_configs.items() if k in args.models}
    
    # Add baseline comparison versions if requested
    if args.include_baseline_comparison:
        additional_configs = {}
        for config_name, config in filtered_configs.items():
            if config['params'].get('rescale_with_baseline', True):
                no_rescale_config = config.copy()
                no_rescale_config['params'] = config['params'].copy()
                no_rescale_config['params']['rescale_with_baseline'] = False
                no_rescale_config['result_name'] = config['result_name'] + '-NoRescale'
                additional_configs[config_name + '_no_rescale'] = no_rescale_config
        
        filtered_configs.update(additional_configs)
    
    # Run all evaluations
    print(f"\nEvaluating BERTScore with models: {list(filtered_configs.keys())}")
    all_results = evaluate_all_bert_score_models(dataset, device, filtered_configs)
    
    # Create and display results
    if all_results:
        df_result = pd.DataFrame(all_results).set_index('Method').round(2)
        
        print("\n" + "=" * 80)
        print("FINAL BERT-SCORE MULTILINGUAL EVALUATION RESULTS")
        print("=" * 80)
        print(df_result)
        
        # Save results to CSV
        output_file = args.output_file
        df_result.to_csv(output_file)
        print(f"\nResults saved to {output_file}")
        
        # Print summary statistics
        print(f"\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Best NoFac AUC: {df_result['NoFac'].max():.2f} ({df_result['NoFac'].idxmax()})")
        print(f"Best NoFac* AUC: {df_result['NoFac*'].max():.2f} ({df_result['NoFac*'].idxmax()})")
        print(f"Best Fac AUC: {df_result['Fac'].max():.2f} ({df_result['Fac'].idxmax()})")
        print(f"Best Pearson: {df_result['Pearson'].max():.2f} ({df_result['Pearson'].idxmax()})")
        print(f"Best Spearman: {df_result['Spearman'].max():.2f} ({df_result['Spearman'].idxmax()})")
        
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
