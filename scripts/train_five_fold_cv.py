import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from model import HATCN
import random
import os
import argparse
import json
from collections import defaultdict

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_metrics(y_true, y_pred):
    """Calculate various metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'tpr': recall_score(y_true, y_pred, pos_label=1),  # ASD
        'tnr': recall_score(y_true, y_pred, pos_label=0)   # non-ASD
    }

def aggregate_segment_predictions(segment_preds, segment_pids, aggregation_method='mean'):
    """Aggregate segment-level predictions to participant-level predictions"""
    unique_pids = np.unique(segment_pids)
    participant_preds = []
    participant_labels = []
    
    for pid in unique_pids:
        # Get predictions for this participant
        pid_mask = segment_pids == pid
        pid_preds = segment_preds[pid_mask]
        
        if aggregation_method == 'mean':
            # Average the segment predictions
            participant_pred = np.mean(pid_preds)
            participant_pred = 1 if participant_pred > 0.5 else 0
        elif aggregation_method == 'majority':
            # Majority vote
            participant_pred = 1 if np.sum(pid_preds) > len(pid_preds) / 2 else 0
        elif aggregation_method == 'max':
            # If any segment predicts ASD, participant is ASD
            participant_pred = 1 if np.any(pid_preds) else 0
        
        participant_preds.append(participant_pred)
        # All segments from same participant have same label
        participant_labels.append(segment_preds[pid_mask][0])
    
    return np.array(participant_preds), np.array(participant_labels)

def train_fold(file_type, fold_idx, device, batch_size=16, epochs=50, patience=5):
    """Train model for a specific fold"""
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx} FOR {file_type.upper()}")
    print(f"{'='*60}")
    
    # Load fold data
    if file_type == 'all':
        # Combine all audio types
        X_train_list, y_train_list, pids_train_list = [], [], []
        X_val_list, y_val_list, pids_val_list = [], [], []
        
        for audio_type in ['reading', 'picture', 'free']:
            fold_dir = f'five_fold_splits/{audio_type}/fold_{fold_idx}'
            if not os.path.exists(fold_dir):
                print(f"Error: {fold_dir} not found. Run extract_features_optimized.py for {audio_type} first.")
                return None
            
            try:
                X_train = np.load(os.path.join(fold_dir, 'features_train.npy'))
                y_train = np.load(os.path.join(fold_dir, 'labels_train.npy'))
                pids_train = np.load(os.path.join(fold_dir, 'participant_ids_train.npy'))
                X_val = np.load(os.path.join(fold_dir, 'features_val.npy'))
                y_val = np.load(os.path.join(fold_dir, 'labels_val.npy'))
                pids_val = np.load(os.path.join(fold_dir, 'participant_ids_val.npy'))
                
                # Add audio type identifier to participant IDs to avoid conflicts
                pids_train = np.array([f"{pid}_{audio_type}" for pid in pids_train])
                pids_val = np.array([f"{pid}_{audio_type}" for pid in pids_val])
                
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                pids_train_list.append(pids_train)
                X_val_list.append(X_val)
                y_val_list.append(y_val)
                pids_val_list.append(pids_val)
                
                print(f"  {audio_type}: {X_train.shape} train, {X_val.shape} val segments")
                
            except FileNotFoundError as e:
                print(f"Error loading {audio_type} fold {fold_idx} data: {e}")
                return None
        
        # Combine all data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        pids_train = np.concatenate(pids_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        pids_val = np.concatenate(pids_val_list, axis=0)
        
        print(f"Combined fold {fold_idx}:")
        print(f"  Train: {X_train.shape} segments from {len(np.unique(pids_train))} participant-audio combinations")
        print(f"  Val: {X_val.shape} segments from {len(np.unique(pids_val))} participant-audio combinations")
        
    else:
        # Single audio type
        fold_dir = f'five_fold_splits/{file_type}/fold_{fold_idx}'
        if not os.path.exists(fold_dir):
            print(f"Error: {fold_dir} not found. Run extract_features_optimized.py first.")
            return None
        
        try:
            X_train = np.load(os.path.join(fold_dir, 'features_train.npy'))
            y_train = np.load(os.path.join(fold_dir, 'labels_train.npy'))
            pids_train = np.load(os.path.join(fold_dir, 'participant_ids_train.npy'))
            X_val = np.load(os.path.join(fold_dir, 'features_val.npy'))
            y_val = np.load(os.path.join(fold_dir, 'labels_val.npy'))
            pids_val = np.load(os.path.join(fold_dir, 'participant_ids_val.npy'))
            
            print(f"Loaded fold {fold_idx}:")
            print(f"  Train: {X_train.shape} segments from {len(np.unique(pids_train))} participants")
            print(f"  Val: {X_val.shape} segments from {len(np.unique(pids_val))} participants")
            
        except FileNotFoundError as e:
            print(f"Error loading fold {fold_idx} data: {e}")
            return None
    
    # Create data loaders
    def make_loader(X, y, batch_size, shuffle):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)
    
    train_loader = make_loader(X_train, y_train, batch_size, True)
    val_loader = make_loader(X_val, y_val, batch_size, False)
    
    # Initialize model
    model = HATCN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    best_f1 = 0
    best_state = None
    patience_counter = 0
    f1_history = []
    smoothing_window = 3
    
    print(f"Training for {epochs} epochs with early stopping (patience={patience})...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds, train_targets = [], []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(yb.cpu().numpy())
        
        train_metrics = get_metrics(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(yb.cpu().numpy())
        
        # Aggregate validation predictions to participant level
        val_preds_participant, val_targets_participant = aggregate_segment_predictions(
            np.array(val_preds), pids_val, 'mean')
        val_metrics = get_metrics(val_targets_participant, val_preds_participant)
        
        # Print progress
        print(f"Epoch {epoch:02d} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | Val F1: {val_metrics['f1']:.3f} | Val Acc: {val_metrics['accuracy']:.3f}")
        
        # Early stopping logic
        f1_history.append(val_metrics['f1'])
        if len(f1_history) >= smoothing_window:
            smoothed_f1 = np.mean(f1_history[-smoothing_window:])
        else:
            smoothed_f1 = val_metrics['f1']
        
        if smoothed_f1 > best_f1:
            best_f1 = smoothed_f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Smoothed Val F1: {best_f1:.3f}")
            break
    
    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
        if file_type == 'all':
            # Create all directory if it doesn't exist
            all_fold_dir = f'five_fold_splits/all/fold_{fold_idx}'
            os.makedirs(all_fold_dir, exist_ok=True)
            model_filename = os.path.join(all_fold_dir, f'best_model_fold_{fold_idx}.pt')
        else:
            model_filename = os.path.join(fold_dir, f'best_model_fold_{fold_idx}.pt')
        torch.save(model.state_dict(), model_filename)
        print(f"Best model saved as {model_filename}")
        
        # Final evaluation on validation set with detailed predictions
        model.eval()
        final_val_preds, final_val_targets = [], []
        final_val_probabilities = []  # Store probabilities for analysis
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probabilities = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                final_val_preds.extend(preds)
                final_val_targets.extend(yb.cpu().numpy())
                final_val_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        final_val_preds = np.array(final_val_preds)
        final_val_targets = np.array(final_val_targets)
        final_val_probabilities = np.array(final_val_probabilities)
        
        # Participant-level metrics and predictions
        final_val_preds_participant, final_val_targets_participant = aggregate_segment_predictions(
            final_val_preds, pids_val, 'mean')
        participant_metrics = get_metrics(final_val_targets_participant, final_val_preds_participant)
        
        # Store all predictions for this fold
        fold_predictions = {
            'segment_level': {
                'predictions': final_val_preds.tolist(),
                'targets': final_val_targets.tolist(),
                'probabilities': final_val_probabilities.tolist(),
                'participant_ids': pids_val.tolist()
            },
            'participant_level': {
                'predictions': final_val_preds_participant.tolist(),
                'targets': final_val_targets_participant.tolist(),
                'participant_ids': list(np.unique(pids_val))
            }
        }
        
        results = {
            'fold': fold_idx,
            'participant_metrics': participant_metrics,
            'best_f1': best_f1,
            'model_path': model_filename,
            'predictions': fold_predictions
        }
        
        return results
    else:
        print("No model was saved.")
        return None

def run_five_fold_cv(file_type, device='cpu', batch_size=16, epochs=50, patience=5):
    """Run 5-fold cross-validation"""
    print(f"\n{'='*80}")
    print(f"5-FOLD CROSS-VALIDATION FOR {file_type.upper()}")
    print(f"{'='*80}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check if splits exist
    if file_type == 'all':
        # Check if all individual audio type splits exist
        for audio_type in ['reading', 'picture', 'free']:
            split_file = f'five_fold_splits/{audio_type}/five_fold_splits.json'
            if not os.path.exists(split_file):
                print(f"Error: {split_file} not found. Run create_five_fold_split.py for {audio_type} first.")
                return
        print("All audio type splits found. Proceeding with combined training...")
    else:
        split_file = f'five_fold_splits/{file_type}/five_fold_splits.json'
        if not os.path.exists(split_file):
            print(f"Error: {split_file} not found. Run create_five_fold_split.py first.")
            return
    
    # Train all folds
    all_results = []
    
    for fold_idx in range(5):
        result = train_fold(file_type, fold_idx, device, batch_size, epochs, patience)
        if result is not None:
            all_results.append(result)
        else:
            print(f"Warning: Fold {fold_idx} failed to complete")
    
    if not all_results:
        print("Error: No folds completed successfully")
        return
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"5-FOLD CROSS-VALIDATION RESULTS FOR {file_type.upper()}")
    print(f"{'='*80}")
    
    # Individual fold results
    print("\nINDIVIDUAL FOLD RESULTS:")
    print("-" * 60)
    
    participant_metrics_agg = defaultdict(list)
    
    for result in all_results:
        fold = result['fold']
        participant_metrics = result['participant_metrics']
        
        print(f"\nFold {fold}:")
        print(f"  Participant Level - Acc: {participant_metrics['accuracy']:.3f}, F1: {participant_metrics['f1']:.3f}, Precision: {participant_metrics['precision']:.3f}, Recall: {participant_metrics['recall']:.3f}")
        
        # Collect for aggregation
        for key, value in participant_metrics.items():
            participant_metrics_agg[key].append(value)
    
    # Aggregated results
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (Mean ± Std)")
    print(f"{'='*60}")
    
    print("\nParticipant Level:")
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'tpr', 'tnr']:
        values = participant_metrics_agg[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Collect all predictions for comprehensive analysis
    all_segment_predictions = []
    all_segment_targets = []
    all_segment_probabilities = []
    all_segment_pids = []
    
    all_participant_predictions = []
    all_participant_targets = []
    all_participant_pids = []
    
    for result in all_results:
        fold_predictions = result['predictions']
        
        # Segment-level predictions
        all_segment_predictions.extend(fold_predictions['segment_level']['predictions'])
        all_segment_targets.extend(fold_predictions['segment_level']['targets'])
        all_segment_probabilities.extend(fold_predictions['segment_level']['probabilities'])
        all_segment_pids.extend(fold_predictions['segment_level']['participant_ids'])
        
        # Participant-level predictions
        all_participant_predictions.extend(fold_predictions['participant_level']['predictions'])
        all_participant_targets.extend(fold_predictions['participant_level']['targets'])
        all_participant_pids.extend(fold_predictions['participant_level']['participant_ids'])
    
    # Create comprehensive results structure
    comprehensive_results = {
        'file_type': file_type,
        'fold_results': all_results,
        'aggregated_participant_metrics': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in participant_metrics_agg.items()},
        'all_predictions': {
            'segment_level': {
                'predictions': all_segment_predictions,
                'targets': all_segment_targets,
                'probabilities': all_segment_probabilities,
                'participant_ids': all_segment_pids
            },
            'participant_level': {
                'predictions': all_participant_predictions,
                'targets': all_participant_targets,
                'participant_ids': all_participant_pids
            }
        }
    }
    
    # Save comprehensive results
    results_file = f'five_fold_splits/{file_type}/cv_results.json'
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Save predictions separately for easy access
    predictions_file = f'five_fold_splits/{file_type}/all_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(comprehensive_results['all_predictions'], f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    print(f"All predictions saved to {predictions_file}")
    
    # Save best overall model
    best_fold_idx = max(all_results, key=lambda x: x['participant_metrics']['f1'])['fold']
    best_model_path = f'five_fold_splits/{file_type}/fold_{best_fold_idx}/best_model_fold_{best_fold_idx}.pt'
    best_overall_model_path = f'five_fold_splits/{file_type}/best_model_{file_type}.pt'
    
    # Copy the best model to a standard location
    import shutil
    shutil.copy2(best_model_path, best_overall_model_path)
    
    # Save best fold info
    best_fold_info = {
        'best_fold': best_fold_idx,
        'best_f1_score': max(all_results, key=lambda x: x['participant_metrics']['f1'])['participant_metrics']['f1'],
        'best_model_path': best_overall_model_path,
        'original_model_path': best_model_path
    }
    
    best_model_info_file = f'five_fold_splits/{file_type}/best_model_info.json'
    with open(best_model_info_file, 'w') as f:
        json.dump(best_fold_info, f, indent=2)
    
    print(f"Best overall model (fold {best_fold_idx}) saved as {best_overall_model_path}")
    print(f"Best model info saved as {best_model_info_file}")
    
    return comprehensive_results

def main():
    parser = argparse.ArgumentParser(description='5-fold cross-validation training')
    parser.add_argument('--file_type', type=str, default='reading', 
                       choices=['reading', 'picture', 'free'],
                       help='Type of audio files to process')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Run cross-validation
    run_five_fold_cv(args.file_type, device, args.batch_size, args.epochs, args.patience)

if __name__ == '__main__':
    main() 