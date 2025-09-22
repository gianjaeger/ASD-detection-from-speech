import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Setup ---
TASKS = ['free', 'picture', 'reading']
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set Times New Roman font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Dictionary to store the confusion matrix for each task
task_confusion_matrices = {}

# --- 1. Generate and Save Individual Plots (Original Behavior) ---
for task in TASKS:
    # Load predictions from the new JSON format
    predictions_file = f'five_fold_splits/{task}/all_predictions.json'
    
    if not os.path.exists(predictions_file):
        print(f"Warning: {predictions_file} not found. Skipping task {task}.")
        continue
    
    try:
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Get participant-level predictions (as these are more meaningful for clinical interpretation)
        y_pred = np.array(predictions_data['participant_level']['predictions'])
        y_true = np.array(predictions_data['participant_level']['targets'])
        
        print(f"Task {task}: {len(y_pred)} participant predictions loaded")
        print(f"  - True labels: {np.sum(y_true)} ASD, {len(y_true) - np.sum(y_true)} Non-ASD")
        print(f"  - Predicted: {np.sum(y_pred)} ASD, {len(y_pred) - np.sum(y_pred)} Non-ASD")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        task_confusion_matrices[task] = cm
        
        # Create individual plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-ASD', 'ASD'])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix - {task.capitalize()}', fontsize=14)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        
        # Update text properties for Times New Roman
        for text in ax.texts:
            text.set_fontsize(12)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{task}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved individual matrix for {task} to {save_path}")
        
    except Exception as e:
        print(f"Error processing task {task}: {e}")
        continue

# --- 2. Generate and Save the Final Combined Plot ---
if task_confusion_matrices:
    print("\nCreating final combined plot with more white space...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    for i, task in enumerate(TASKS):
        ax = axes[i]
        cm = task_confusion_matrices.get(task)
        
        if cm is None:
            ax.text(0.5, 0.5, f'{task.capitalize()}\n(No data)', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-ASD', 'ASD'])
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

        ax.set_title('')
        ax.set_xlabel('Predicted label', fontsize=18)
        ax.tick_params(axis='x', labelsize=15)

        # Update text properties for Times New Roman
        for text in ax.texts:
            text.set_fontsize(20)
        
        ax.set_ylabel('')

    axes[0].set_ylabel('True label', fontsize=18)
    axes[0].tick_params(axis='y', labelsize=15)

    # Adjust layout with more horizontal space between plots
    plt.tight_layout(w_pad=4.0)
    
    combined_save_path = os.path.join(RESULTS_DIR, 'confusion_matrix_final_combined_spaced.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🎉 Successfully saved final combined matrix to {combined_save_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CONFUSION MATRIX SUMMARY")
    print("="*60)
    for task in TASKS:
        if task in task_confusion_matrices:
            cm = task_confusion_matrices[task]
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"\n{task.upper()}:")
            print(f"  True Negatives: {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  True Positives: {tp}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Sensitivity (TPR): {sensitivity:.3f}")
            print(f"  Specificity (TNR): {specificity:.3f}")
            print(f"  Precision: {precision:.3f}")
else:
    print("No confusion matrices were created. Check if prediction files exist.")