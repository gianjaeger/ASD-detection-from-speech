import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def load_demographics_data():
    """Load all demographic JSON files and extract ASD severity information"""
    demographics = {}
    
    # Load ASD participants
    asd_dir = "step3-normalized_data/ASD speech segments"
    if os.path.exists(asd_dir):
        for participant_id in os.listdir(asd_dir):
            participant_dir = os.path.join(asd_dir, participant_id)
            if os.path.isdir(participant_dir):
                # Look for JSON file
                json_files = [f for f in os.listdir(participant_dir) if f.endswith('.json')]
                if json_files:
                    json_file = os.path.join(participant_dir, json_files[0])
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            if 'autism_severity' in data:
                                demographics[participant_id] = {
                                    'severity': data['autism_severity'],
                                    'group': 'ASD',
                                    'true_label': 1  # ASD = 1
                                }
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
    
    # Load non-ASD participants
    non_asd_dir = "step3-normalized_data/non-ASD Speech segments "
    if os.path.exists(non_asd_dir):
        for participant_id in os.listdir(non_asd_dir):
            participant_dir = os.path.join(non_asd_dir, participant_id)
            if os.path.isdir(participant_dir):
                # Look for JSON file
                json_files = [f for f in os.listdir(participant_dir) if f.endswith('.json')]
                if json_files:
                    json_file = os.path.join(participant_dir, json_files[0])
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            demographics[participant_id] = {
                                'severity': 'non-ASD',
                                'group': 'non-ASD',
                                'true_label': 0  # Non-ASD = 0
                            }
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
    
    return demographics

def load_predictions(task):
    """Load predictions for a specific task"""
    predictions_file = f'five_fold_splits/{task}/all_predictions.json'
    
    if not os.path.exists(predictions_file):
        print(f"Warning: {predictions_file} not found.")
        return None
    
    try:
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Get participant-level predictions
        y_pred = np.array(predictions_data['participant_level']['predictions'])
        participant_ids = predictions_data['participant_level']['participant_ids']
        
        return {
            'predictions': y_pred,
            'participant_ids': participant_ids
        }
    except Exception as e:
        print(f"Error loading predictions for {task}: {e}")
        return None

def analyze_predictions_by_severity():
    """Analyze prediction accuracy by ASD severity levels using demographics as ground truth"""
    
    # Load demographics data
    print("Loading demographic data...")
    demographics = load_demographics_data()
    print(f"Loaded demographics for {len(demographics)} participants")
    
    # Analyze each task
    tasks = ['free', 'picture', 'reading']
    results = {}
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"ANALYZING {task.upper()} TASK")
        print(f"{'='*60}")
        
        # Load predictions
        pred_data = load_predictions(task)
        if pred_data is None:
            print(f"Skipping {task} - no prediction data available")
            continue
        
        # Create mapping from participant IDs to predictions
        id_to_pred = {}
        
        for i, pid in enumerate(pred_data['participant_ids']):
            # Clean participant ID (remove audio type suffix if present)
            clean_pid = pid.split('_')[0] if '_' in pid else pid
            id_to_pred[clean_pid] = pred_data['predictions'][i]
        
        # Group by severity using demographics as ground truth
        severity_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'participants': []})
        
        for pid, demo_info in demographics.items():
            if pid in id_to_pred:
                pred = id_to_pred[pid]
                true = demo_info['true_label']  # Use demographics as ground truth
                severity = demo_info['severity']
                
                is_correct = (pred == true)
                severity_results[severity]['total'] += 1
                if is_correct:
                    severity_results[severity]['correct'] += 1
                
                severity_results[severity]['participants'].append({
                    'pid': pid,
                    'prediction': pred,
                    'true': true,
                    'correct': is_correct
                })
        
        # Calculate accuracy for each severity level
        task_results = {}
        print(f"\nResults for {task.upper()} task:")
        print("-" * 40)
        
        for severity, data in severity_results.items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                task_results[severity] = {
                    'accuracy': accuracy,
                    'correct': data['correct'],
                    'total': data['total'],
                    'participants': data['participants']
                }
                
                print(f"{severity.upper()}: {accuracy:.1f}% ({data['correct']}/{data['total']})")
                
                # Print individual participant results for debugging
                for participant in data['participants']:
                    status = "✓" if participant['correct'] else "✗"
                    print(f"  {status} {participant['pid']}: Predicted {participant['prediction']}, True {participant['true']}")
        
        results[task] = task_results
    
    return results

def create_summary_report(results):
    """Create a comprehensive summary report"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PREDICTION ACCURACY BY SEVERITY")
    print(f"{'='*80}")
    
    # Create summary table
    all_severities = set()
    for task_results in results.values():
        all_severities.update(task_results.keys())
    
    # Sort severities in logical order
    severity_order = ['non-ASD', 'mild', 'moderate', 'severe', 'not_sure']
    ordered_severities = [s for s in severity_order if s in all_severities]
    
    print(f"\n{'Task':<12} {'Severity':<12} {'Accuracy':<10} {'Correct/Total':<15}")
    print("-" * 60)
    
    for task in ['free', 'picture', 'reading']:
        if task in results:
            for severity in ordered_severities:
                if severity in results[task]:
                    data = results[task][severity]
                    print(f"{task.upper():<12} {severity.upper():<12} {data['accuracy']:<10.1f}% {data['correct']}/{data['total']:<15}")
    
    # Calculate overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    for task in ['free', 'picture', 'reading']:
        if task in results:
            print(f"\n{task.upper()} TASK:")
            total_correct = 0
            total_participants = 0
            
            for severity, data in results[task].items():
                total_correct += data['correct']
                total_participants += data['total']
            
            if total_participants > 0:
                overall_accuracy = (total_correct / total_participants) * 100
                print(f"  Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_participants})")
    
    # Save detailed results to JSON
    output_file = 'severity_analysis_results_corrected.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to {output_file}")

def main():
    """Main function to run the severity analysis"""
    print("ASD Severity Prediction Analysis (Corrected)")
    print("=" * 50)
    
    # Run analysis
    results = analyze_predictions_by_severity()
    
    # Create summary report
    create_summary_report(results)
    
    print(f"\nAnalysis complete! Check severity_analysis_results_corrected.json for detailed results.")

if __name__ == "__main__":
    main() 