#!/usr/bin/env python3
"""
Evaluation script for all ViSelfCheck methods based on the experiment notebook.

This script evaluates all methods available in the ViSelfCheck service
using the Vietnamese Hallucination dataset.
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from typing import Dict, List, Any

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


def evaluate_method(method_name: str, dataset, device, **method_kwargs):
    """
    Evaluate a specific method using the unified ViSelfCheck interface.
    
    Args:
        method_name: Name of the method to evaluate
        dataset: Dataset to evaluate on
        device: Device to use for computation
        **method_kwargs: Method-specific parameters
        
    Returns:
        Dictionary with method scores
    """
    print("=" * 50)
    print(f"Evaluating {method_name} method...")
    
    # Initialize the checker with the specified method
    checker = ViSelfCheck(method=method_name, device=device, **method_kwargs)
    scores = {}
    
    num_datapoints = len(dataset)
    with tqdm(total=num_datapoints, desc=f'SelfCheck-{method_name}') as pbar:
        for i, datapoint in enumerate(dataset):
            # For some methods, we need to pass the passage parameter
            predict_kwargs = {}
            if method_name in ['mqag', 'ngram']:
                predict_kwargs['passage'] = datapoint['gemini_text']
            
            # Special handling for ngram smoothing
            if method_name == 'ngram':
                predict_kwargs['smoothing_pseudo_count'] = 0
            
            selfcheck_scores_ = checker.predict(
                sentences=datapoint['gemini_sentences'],
                sampled_passages=datapoint['gemini_text_samples'],
                **predict_kwargs
            )
            scores[i] = selfcheck_scores_
            pbar.update(1)
    
    return scores


def evaluate_all_methods(dataset, device, method_configs):
    """
    Evaluate all methods specified in method_configs.
    
    Args:
        dataset: Dataset to evaluate on
        device: Device to use for computation
        method_configs: Dictionary of method configurations
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for method_name, config in method_configs.items():
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING {method_name.upper()} METHOD")
            print(f"{'='*60}")
            
            # Extract method parameters
            method_params = config.get('params', {})
            result_names = config.get('result_names', [method_name])
            
            # Standard method evaluation for all methods
            scores = evaluate_method(method_name, dataset, device, **method_params)
            result_name = result_names[0] if result_names else method_name
            
            # For N-gram methods, we need to provide passage-level scores
            if 'gram' in method_name.lower():
                # Calculate passage-level scores by averaging sentence-level scores
                passage_scores = {}
                for idx, sent_scores in scores.items():
                    passage_scores[idx] = np.mean(sent_scores)
                result = result_collect(scores, dataset, result_name, passage_scores)
            else:
                result = result_collect(scores, dataset, result_name)
                
            all_results.append(result)
                
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def test_ngram_only(dataset, ngram_range=(1, 2)):
    """Test only N-gram method for quick validation."""
    print("=" * 50)
    print(f"Testing N-gram method (n={ngram_range[0]} to {ngram_range[1]})...")
    
    # Use only first few datapoints for quick testing
    test_size = min(6, len(dataset))
    test_dataset = [dataset[i] for i in range(test_size)]
    
    results = []
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        print(f"\nTesting {n}-gram...")
        
        try:
            # Use unified interface
            checker = ViSelfCheck(method='ngram', n=n, lowercase=True)
            
            scores = {}
            for i, datapoint in enumerate(test_dataset):
                print(f"  Processing datapoint {i+1}/{len(test_dataset)}...")
                
                selfcheck_scores_ = checker.predict(
                    sentences=datapoint['gemini_sentences'],
                    sampled_passages=datapoint['gemini_text_samples'],
                )
                
                # Check for None values and handle them
                if selfcheck_scores_ is None:
                    print(f"    Warning: Got None result for datapoint {i+1}")
                    selfcheck_scores_ = [0.0] * len(datapoint['gemini_sentences'])
                elif any(score is None for score in selfcheck_scores_):
                    print(f"    Warning: Got None scores in result for datapoint {i+1}")
                    selfcheck_scores_ = [0.0 if score is None else score for score in selfcheck_scores_]
                
                scores[i] = selfcheck_scores_
            
            # Create a small dataset for result collection
            from datasets import Dataset
            test_dataset_obj = Dataset.from_list(test_dataset)
            
            # Collect results
            # For N-gram methods, we need to provide passage-level scores
            if 'gram' in f'{n}-gram'.lower():
                # Calculate passage-level scores by averaging sentence-level scores
                passage_scores = {}
                for idx, sent_scores in scores.items():
                    passage_scores[idx] = np.mean(sent_scores)
                result = result_collect(scores, test_dataset_obj, f'{n}-gram', passage_scores)
            else:
                result = result_collect(scores, test_dataset_obj, f'{n}-gram')
                
            results.append(result)
            
            print(f"  {n}-gram results: {result}")
            
        except Exception as e:
            print(f"  Error testing {n}-gram: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Main function to run all evaluations."""
    parser = argparse.ArgumentParser(description='Evaluate ViSelfCheck methods')
    parser.add_argument('--dataset-path', default='Vietnamese_Hallucination', 
                       help='Path to the dataset directory')
    parser.add_argument('--ngram-only', action='store_true',
                       help='Run only N-gram evaluation for testing')
    parser.add_argument('--ngram-n', type=int, default=1,
                       help='N-gram size (default: 1 for unigram)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--methods', nargs='+', 
                       default=['bert_score', 'nli', 'mqag', 'ngram', 'prompt', 'hybrid'],
                       help='Methods to evaluate')
    parser.add_argument('--output-file', default='evaluation_results.csv',
                       help='Output CSV file name (default: evaluation_results.csv)')
    parser.add_argument('--api-key', default=None,
                       help='API key for prompt and hybrid methods')
    parser.add_argument('--prompt-model', default='gpt-3.5-turbo',
                       help='Model to use for prompt method (default: gpt-3.5-turbo)')
    parser.add_argument('--prompt-client', default='openai',
                       help='Client type for prompt method (default: openai)')
    
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
    
    # If ngram-only mode, run only N-gram tests
    if args.ngram_only:
        results = test_ngram_only(dataset, (args.ngram_n, 5))
        if results:
            df_result = pd.DataFrame(results).set_index('Method').round(2)
            print("\n" + "=" * 50)
            print("N-gram Test Results:")
            print("=" * 50)
            print(df_result)
            
            # Save test results to CSV
            test_output_file = f'ngram_test_results_{args.ngram_n}.csv'
            df_result.to_csv(test_output_file)
            print(f"\nTest results saved to {test_output_file}")
        return
    
    # Define method configurations
    method_configs = {
        'bert_score': {
            'params': {'rescale_with_baseline': True},
            'result_names': ['BERTScore']
        },
        'nli': {
            'params': {},
            'result_names': ['NLI']
        },
        'mqag': {
            'params': {},
            'result_names': ['MQAG']
        },
        'ngram': {
            'params': {'n': args.ngram_n},
            'result_names': [f'{args.ngram_n}-gram']
        },
        'prompt': {
            'params': {
                'client_type': args.prompt_client,
                'model': args.prompt_model,
                'api_key': args.api_key
            },
            'result_names': ['Prompt']
        },
        'hybrid': {
            'params': {
                'api_key': args.api_key,
                'llm_model': args.prompt_model
            },
            'result_names': ['Hybrid']
        }
    }
    
    # Filter methods based on user selection
    filtered_configs = {k: v for k, v in method_configs.items() if k in args.methods}
    
    # Run all evaluations
    print(f"\nEvaluating methods: {list(filtered_configs.keys())}")
    all_results = evaluate_all_methods(dataset, device, filtered_configs)
    
    # Create and display results
    if all_results:
        df_result = pd.DataFrame(all_results).set_index('Method').round(2)
        
        print("\n" + "=" * 60)
        print("FINAL EVALUATION RESULTS")
        print("=" * 60)
        print(df_result)
        
        # Save results to CSV
        output_file = args.output_file
        df_result.to_csv(output_file)
        print(f"\nResults saved to {output_file}")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
