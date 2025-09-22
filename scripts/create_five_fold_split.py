import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse

def create_five_fold_split(file_type='reading'):
    """
    Create 5-fold cross-validation splits at the participant level.
    Ensures that all segments from the same participant stay in the same fold.
    """
    print(f"Creating 5-fold cross-validation splits for {file_type} data...")
    
    # Load the file list
    if not os.path.exists('filelist_enhanced.csv'):
        print("Error: filelist_enhanced.csv not found. Run prepare_data_enhanced.py first.")
        return
    
    # Read the file list
    df = pd.read_csv('filelist_enhanced.csv')
    
    # Filter for the specified file type
    df_filtered = df[df['snippet_type'] == file_type].copy()
    
    if len(df_filtered) == 0:
        print(f"Error: No {file_type} files found in filelist_enhanced.csv")
        return
    
    # Get unique participants and their labels
    participants = df_filtered[['participant_id', 'label']].drop_duplicates()
    participant_ids = participants['participant_id'].values
    participant_labels = participants['label'].values
    
    print(f"Found {len(participants)} participants with {file_type} data")
    print(f"ASD participants: {np.sum(participant_labels == 1)}")
    print(f"Non-ASD participants: {np.sum(participant_labels == 0)}")
    
    # Create 5-fold splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create output directory
    output_dir = f'five_fold_splits/{file_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate splits
    fold_data = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(participant_ids, participant_labels)):
        train_participants = participant_ids[train_idx]
        val_participants = participant_ids[val_idx]
        
        # Get all files for these participants
        train_files = df_filtered[df_filtered['participant_id'].isin(train_participants)]
        val_files = df_filtered[df_filtered['participant_id'].isin(val_participants)]
        
        fold_data[f'fold_{fold_idx}'] = {
            'train_participants': train_participants.tolist(),
            'val_participants': val_participants.tolist(),
            'train_files': train_files.to_dict('records'),
            'val_files': val_files.to_dict('records')
        }
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_participants)} participants ({len(train_files)} files)")
        print(f"  Val: {len(val_participants)} participants ({len(val_files)} files)")
        print(f"  Train ASD: {np.sum(participant_labels[train_idx] == 1)}, Non-ASD: {np.sum(participant_labels[train_idx] == 0)}")
        print(f"  Val ASD: {np.sum(participant_labels[val_idx] == 1)}, Non-ASD: {np.sum(participant_labels[val_idx] == 0)}")
        print()
    
    # Save the splits
    split_file = os.path.join(output_dir, 'five_fold_splits.json')
    with open(split_file, 'w') as f:
        json.dump(fold_data, f, indent=2)
    
    # Save participant summary for this file type
    participant_summary = participants.copy()
    participant_summary.to_csv(os.path.join(output_dir, 'participant_summary.csv'), index=False)
    
    print(f"Saved 5-fold splits to {split_file}")
    print(f"Saved participant summary to {os.path.join(output_dir, 'participant_summary.csv')}")
    
    return fold_data

def create_all_five_fold_splits():
    """
    Create 5-fold cross-validation splits for all audio types at once.
    """
    print("Creating 5-fold cross-validation splits for all audio types...")
    
    # Load the file list
    if not os.path.exists('filelist_enhanced.csv'):
        print("Error: filelist_enhanced.csv not found. Run prepare_data_enhanced.py first.")
        return
    
    # Read the file list
    df = pd.read_csv('filelist_enhanced.csv')
    
    # Create splits for each audio type
    audio_types = ['reading', 'picture', 'free']
    
    for file_type in audio_types:
        print(f"\n{'='*50}")
        create_five_fold_split(file_type)
    
    print(f"\n{'='*50}")
    print("✅ All five-fold splits created successfully!")
    print("Created splits for: reading, picture, free")
    print("You can now run feature extraction for each type:")
    print("  python extract_features_five_fold_simple.py --file_type reading")
    print("  python extract_features_five_fold_simple.py --file_type picture")
    print("  python extract_features_five_fold_simple.py --file_type free")

def main():
    parser = argparse.ArgumentParser(description='Create 5-fold cross-validation splits')
    parser.add_argument('--file_type', type=str, default='all', 
                       choices=['reading', 'picture', 'free', 'all'],
                       help='Type of audio files to process (or "all" for all types)')
    
    args = parser.parse_args()
    
    if args.file_type == 'all':
        create_all_five_fold_splits()
    else:
        create_five_fold_split(args.file_type)

if __name__ == '__main__':
    main() 