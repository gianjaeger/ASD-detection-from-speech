import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.metrics import confusion_matrix

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

def create_task_correlation_data():
    """Create a dataset with prediction accuracy for each participant across all tasks"""
    
    # Load demographics
    demographics = load_demographics_data()
    
    # Load predictions for all tasks
    tasks = ['free', 'picture', 'reading']
    task_predictions = {}
    
    for task in tasks:
        pred_data = load_predictions(task)
        if pred_data is not None:
            # Create mapping from participant IDs to predictions
            id_to_pred = {}
            for i, pid in enumerate(pred_data['participant_ids']):
                # Clean participant ID (remove audio type suffix if present)
                clean_pid = pid.split('_')[0] if '_' in pid else pid
                id_to_pred[clean_pid] = pred_data['predictions'][i]
            task_predictions[task] = id_to_pred
    
    # Create correlation dataset
    correlation_data = []
    
    for pid, demo_info in demographics.items():
        participant_data = {
            'participant_id': pid,
            'true_label': demo_info['true_label'],
            'group': demo_info['group'],
            'severity': demo_info['severity']
        }
        
        # Add prediction accuracy for each task
        for task in tasks:
            if task in task_predictions and pid in task_predictions[task]:
                pred = task_predictions[task][pid]
                true = demo_info['true_label']
                is_correct = (pred == true)
                participant_data[f'{task}_correct'] = is_correct
                participant_data[f'{task}_prediction'] = pred
            else:
                participant_data[f'{task}_correct'] = None
                participant_data[f'{task}_prediction'] = None
        
        correlation_data.append(participant_data)
    
    return pd.DataFrame(correlation_data)

def analyze_task_correlations(df):
    """Analyze correlations between task performances"""
    
    print("TASK CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Filter out participants with missing data
    df_clean = df.dropna(subset=['free_correct', 'picture_correct', 'reading_correct'])
    print(f"Participants with complete data: {len(df_clean)}/{len(df)}")
    
    # Convert boolean to numeric for correlation analysis
    for task in ['free', 'picture', 'reading']:
        df_clean[f'{task}_correct'] = df_clean[f'{task}_correct'].astype(int)
    
    # 1. Overall correlation between tasks
    print("\n1. OVERALL CORRELATION BETWEEN TASKS")
    print("-" * 40)
    
    tasks = ['free', 'picture', 'reading']
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks[i+1:], i+1):
            corr_pearson, p_pearson = pearsonr(df_clean[f'{task1}_correct'], df_clean[f'{task2}_correct'])
            corr_spearman, p_spearman = spearmanr(df_clean[f'{task1}_correct'], df_clean[f'{task2}_correct'])
            
            print(f"{task1.upper()} vs {task2.upper()}:")
            print(f"  Pearson r: {corr_pearson:.3f} (p={p_pearson:.3f})")
            print(f"  Spearman ρ: {corr_spearman:.3f} (p={p_spearman:.3f})")
            print()
    
    # 2. Correlation by group (ASD vs Non-ASD)
    print("\n2. CORRELATION BY GROUP")
    print("-" * 40)
    
    for group in ['ASD', 'non-ASD']:
        group_df = df_clean[df_clean['group'] == group]
        print(f"\n{group.upper()} participants (n={len(group_df)}):")
        
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                corr_pearson, p_pearson = pearsonr(group_df[f'{task1}_correct'], group_df[f'{task2}_correct'])
                corr_spearman, p_spearman = spearmanr(group_df[f'{task1}_correct'], group_df[f'{task2}_correct'])
                
                print(f"  {task1.upper()} vs {task2.upper()}:")
                print(f"    Pearson r: {corr_pearson:.3f} (p={p_pearson:.3f})")
                print(f"    Spearman ρ: {corr_spearman:.3f} (p={p_spearman:.3f})")
    
    # 3. Cross-task consistency analysis
    print("\n3. CROSS-TASK CONSISTENCY ANALYSIS")
    print("-" * 40)
    
    # Calculate how many participants are correct in 0, 1, 2, or 3 tasks
    df_clean['correct_count'] = df_clean[['free_correct', 'picture_correct', 'reading_correct']].sum(axis=1)
    
    print("Participants correct in X tasks:")
    for i in range(4):
        count = (df_clean['correct_count'] == i).sum()
        percentage = (count / len(df_clean)) * 100
        print(f"  {i} tasks: {count} participants ({percentage:.1f}%)")
    
    # By group
    for group in ['ASD', 'non-ASD']:
        group_df = df_clean[df_clean['group'] == group]
        print(f"\n{group.upper()} participants - correct in X tasks:")
        for i in range(4):
            count = (group_df['correct_count'] == i).sum()
            percentage = (count / len(group_df)) * 100
            print(f"  {i} tasks: {count} participants ({percentage:.1f}%)")
    
    # 4. Task-specific analysis
    print("\n4. TASK-SPECIFIC ANALYSIS")
    print("-" * 40)
    
    for task in tasks:
        task_correct = df_clean[f'{task}_correct'].sum()
        task_total = len(df_clean)
        task_accuracy = (task_correct / task_total) * 100
        
        print(f"\n{task.upper()} TASK:")
        print(f"  Overall accuracy: {task_accuracy:.1f}% ({task_correct}/{task_total})")
        
        # By group
        for group in ['ASD', 'non-ASD']:
            group_df = df_clean[df_clean['group'] == group]
            group_correct = group_df[f'{task}_correct'].sum()
            group_total = len(group_df)
            group_accuracy = (group_correct / group_total) * 100
            print(f"  {group}: {group_accuracy:.1f}% ({group_correct}/{group_total})")
    
    return df_clean

def create_correlation_visualizations(df_clean):
    """Create visualizations for task correlations"""
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 1. Correlation heatmap
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall correlation heatmap
    tasks = ['free', 'picture', 'reading']
    corr_matrix = df_clean[[f'{task}_correct' for task in tasks]].corr()
    corr_matrix.columns = [task.upper() for task in tasks]
    corr_matrix.index = [task.upper() for task in tasks]
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0,0], cbar_kws={'shrink': 0.8})
    axes[0,0].set_title('Overall Task Correlation', fontsize=14, fontweight='bold')
    
    # ASD group correlation heatmap
    asd_df = df_clean[df_clean['group'] == 'ASD']
    if len(asd_df) > 0:
        asd_corr_matrix = asd_df[[f'{task}_correct' for task in tasks]].corr()
        asd_corr_matrix.columns = [task.upper() for task in tasks]
        asd_corr_matrix.index = [task.upper() for task in tasks]
        
        sns.heatmap(asd_corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                    square=True, ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('ASD Group Task Correlation', fontsize=14, fontweight='bold')
    
    # Non-ASD group correlation heatmap
    non_asd_df = df_clean[df_clean['group'] == 'non-ASD']
    if len(non_asd_df) > 0:
        non_asd_corr_matrix = non_asd_df[[f'{task}_correct' for task in tasks]].corr()
        non_asd_corr_matrix.columns = [task.upper() for task in tasks]
        non_asd_corr_matrix.index = [task.upper() for task in tasks]
        
        sns.heatmap(non_asd_corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                    square=True, ax=axes[1,0], cbar_kws={'shrink': 0.8})
        axes[1,0].set_title('Non-ASD Group Task Correlation', fontsize=14, fontweight='bold')
    
    # 2. Cross-task consistency histogram
    axes[1,1].hist(df_clean['correct_count'], bins=4, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Number of Tasks Correct')
    axes[1,1].set_ylabel('Number of Participants')
    axes[1,1].set_title('Cross-Task Consistency Distribution', fontsize=14, fontweight='bold')
    axes[1,1].set_xticks([0, 1, 2, 3])
    
    plt.tight_layout()
    plt.savefig('task_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrelation visualization saved as 'task_correlation_analysis.png'")

def main():
    """Main function to run the task correlation analysis"""
    print("TASK CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Create correlation dataset
    df = create_task_correlation_data()
    
    # Analyze correlations
    df_clean = analyze_task_correlations(df)
    
    # Create visualizations
    create_correlation_visualizations(df_clean)
    
    # Save detailed results
    output_file = 'task_correlation_results.json'
    results = {
        'participant_data': df_clean.to_dict('records'),
        'summary_stats': {
            'total_participants': len(df),
            'complete_data_participants': len(df_clean),
            'asd_participants': len(df_clean[df_clean['group'] == 'ASD']),
            'non_asd_participants': len(df_clean[df_clean['group'] == 'non-ASD'])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to {output_file}")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main() 