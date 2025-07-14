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
    rescale_with_baseline: bool = False,
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
    
    # Initialize BERTScore checker with specific model
    bert_score_params = {
        'rescale_with_baseline': rescale_with_baseline,
        'model_type': model_name,
        'device': device
    }
    
    
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



def evaluate_all_models(model_names, dataset, device):
    """
    Evaluate all models and collect results.
    
    Args:
        model_names: List of model names to evaluate
        dataset: Dataset to evaluate on
        device: Device to use for computation
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Evaluate the model
            scores = evaluate_bert_score_model(
                model_name=model_name,
                dataset=dataset,
                device=device,
                rescale_with_baseline=False
            )
            
            if scores is not None:
                # Create result name from model name
       
                # Collect results
                result = result_collect(scores, dataset, model_name)
                all_results.append(result)
                
                print(f"Results for {model_name}:")
                for key, value in result.items():
                    if key != 'Method':
                        print(f"  {key}: {value:.2f}")
            else:
                print(f"Failed to evaluate {model_name}")
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


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
                       help='Model names to evaluate (e.g.,bert-base-multilingual-cased)')
    parser.add_argument('--output-file', default='bert_score_results.csv',
                       help='Output CSV file name')
    
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
        test_models = ['bert-base-multilingual-cased', 'microsoft/deberta-xlarge-mnli']
        test_size = min(args.num_samples, len(dataset))
        test_dataset = [dataset[i] for i in range(test_size)]
        
        all_results = evaluate_all_models(test_models, test_dataset, device)
        if all_results:
            df_result = pd.DataFrame(all_results).set_index('Method').round(2)
            test_output_file = args.output_file
            df_result.to_csv(test_output_file)
            print(f"\nQuick test results saved to {test_output_file}")
        return
    
    # Get model names to evaluate
    if args.models:
        model_names = args.models
        print(f"\nEvaluating BERTScore with specified models: {model_names}")
    else:
        # Default models if none specified
        model_names = [
            'vinai/phobert-base-v2',
            'bert-base-multilingual-cased', 
            'vinai/phobert-large',
            'microsoft/deberta-xlarge-mnli',
            'microsoft/deberta-v2-xxlarge-mnli',
            'microsoft/deberta-xlarge'
        ]
        print(f"\nEvaluating BERTScore with default models: {model_names}")
    
    # Run all evaluations with specified models
    all_results = evaluate_all_models(model_names, dataset, device)
    
    # Create and display results
    if all_results:
        df_result = pd.DataFrame(all_results).set_index('Method').round(2)
        
        print("\n" + "=" * 80)
        print("FINAL BERT-SCORE EVALUATION RESULTS")
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
