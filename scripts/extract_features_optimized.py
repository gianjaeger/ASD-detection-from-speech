import os
import json
import numpy as np
import librosa
import parselmouth
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm

# Feature extraction parameters
SAMPLE_RATE = 16000
FRAME_LENGTH = int(0.03 * SAMPLE_RATE)  # 30 ms
HOP_LENGTH = int(0.01 * SAMPLE_RATE)   # 10 ms
N_MFCC = 13
N_DOCC = 13

# Segmentation parameters - using 4 seconds like the original for speed
SEGMENT_DURATION = 4.0  # Changed from 3.0 to 4.0 for speed
SEGMENT_OVERLAP = 1.0   # 1 second overlap between segments
SEGMENT_LENGTH = int(SEGMENT_DURATION * SAMPLE_RATE)
SEGMENT_HOP = int((SEGMENT_DURATION - SEGMENT_OVERLAP) * SAMPLE_RATE)

def extract_librosa_features(y, sr):
    energy = np.log(np.sum(librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)**2, axis=0) + 1e-6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    docc = librosa.feature.delta(mfcc, width=5, order=1, mode='mirror')
    docc_delta = librosa.feature.delta(mfcc, width=5, order=2, mode='mirror')
    return energy, mfcc, mfcc_delta, mfcc_delta2, docc, docc_delta

def extract_parselmouth_features(y, sr):
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        pitch = snd.to_pitch(time_step=HOP_LENGTH/sr)
        f0 = pitch.selected_array['frequency']
        f0 = np.nan_to_num(f0)
    except Exception:
        f0 = np.zeros(int(np.ceil(len(y)/HOP_LENGTH)))
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        periods = [parselmouth.praat.call(point_process, "Get interval duration", i) for i in range(1, point_process.get_number_of_points())]
        jitter_local = np.zeros_like(f0)
        if len(periods) > 1:
            mean_period = np.mean(periods)
            for i in range(1, len(periods)):
                jitter_local[i] = abs(periods[i] - periods[i-1]) / mean_period if mean_period > 0 else 0
        n_frames = len(f0)
        if len(jitter_local) < n_frames:
            jitter_local = np.pad(jitter_local, (0, n_frames - len(jitter_local)), mode='constant')
        elif len(jitter_local) > n_frames:
            jitter_local = jitter_local[:n_frames]
        jitter = jitter_local
    except Exception:
        jitter = np.zeros_like(f0)
    try:
        formant = snd.to_formant_burg(time_step=HOP_LENGTH/sr)
        n_frames = int(np.ceil(len(y)/HOP_LENGTH))
        f1, f2, f3 = [], [], []
        for i in range(n_frames):
            t = i * HOP_LENGTH / sr
            f1.append(formant.get_value_at_time(1, t) or 0)
            f2.append(formant.get_value_at_time(2, t) or 0)
            f3.append(formant.get_value_at_time(3, t) or 0)
        f1, f2, f3 = np.array(f1), np.array(f2), np.array(f3)
    except Exception:
        f1 = f2 = f3 = np.zeros(int(np.ceil(len(y)/HOP_LENGTH)))
    return f0, f1, f2, f3, jitter

def extract_features_for_segment(segment_audio, sr):
    energy, mfcc, mfcc_delta, mfcc_delta2, docc, docc_delta = extract_librosa_features(segment_audio, sr)
    f0, f1, f2, f3, jitter = extract_parselmouth_features(segment_audio, sr)
    n_frames = mfcc.shape[1]
    def fix(x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] < n_frames:
            pad_width = n_frames - x.shape[0]
            x = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')
        elif x.shape[0] > n_frames:
            x = x[:n_frames]
        return x
    features = np.hstack([
        fix(f0),                  # 1
        fix(energy),              # 1
        fix(f1), fix(f2), fix(f3),# 3
        fix(jitter),              # 1
        fix(mfcc.T),              # 13
        fix(mfcc_delta.T),        # 13
        fix(mfcc_delta2.T),       # 13
        fix(docc.T),              # 13
        fix(docc_delta.T)         # 13
    ])
    if features.shape[1] != 84:
        features = features[:, :84] if features.shape[1] > 84 else np.pad(features, ((0, 0), (0, 84 - features.shape[1])), mode='constant')
    return features

def segment_audio(y, sr, segment_length, segment_hop):
    segments = []
    start = 0
    while start + segment_length <= len(y):
        segment = y[start:start + segment_length]
        segments.append(segment)
        start += segment_hop
    return segments

def extract_features_for_file(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"[ERROR] Could not load {filepath}: {e}")
        return None
    segments = segment_audio(y, sr, SEGMENT_LENGTH, SEGMENT_HOP)
    if not segments:
        print(f"[WARNING] No segments extracted from {filepath}")
        return None
    segment_features = []
    for i, segment in enumerate(segments):
        features = extract_features_for_segment(segment, sr)
        segment_features.append(features)
    return np.array(segment_features)  # [n_segments, n_frames, 84]

def extract_features_for_all_folds(file_type):
    """Extract features for all 5 folds at once - optimized version"""
    print(f"Extracting features for {file_type} - all 5 folds...")
    
    # Load the fold splits
    split_file = f'five_fold_splits/{file_type}/five_fold_splits.json'
    if not os.path.exists(split_file):
        print(f"Error: {split_file} not found. Run create_five_fold_split.py first.")
        return
    
    with open(split_file, 'r') as f:
        fold_data = json.load(f)
    
    # Get all unique files across all folds
    all_files = set()
    for fold_key in fold_data.keys():
        fold_info = fold_data[fold_key]
        for file_info in fold_info['train_files'] + fold_info['val_files']:
            all_files.add(file_info['filepath'])
    
    all_files = list(all_files)
    print(f"Found {len(all_files)} unique files to process")
    
    # Extract features for all files once
    print("Extracting features for all files...")
    file_features = {}
    successful_files = 0
    
    for filepath in tqdm(all_files, desc="Processing files"):
        segment_features = extract_features_for_file(filepath)
        if segment_features is not None:
            file_features[filepath] = segment_features
            successful_files += 1
    
    print(f"Successfully processed {successful_files}/{len(all_files)} files ({successful_files/len(all_files)*100:.1f}%)")
    
    if successful_files == 0:
        print("Error: No files were successfully processed. Check the error messages above.")
        return
    
    # Process each fold
    for fold_idx in range(5):
        fold_key = f'fold_{fold_idx}'
        fold_info = fold_data[fold_key]
        
        print(f"\nProcessing fold {fold_idx}...")
        
        # Create output directory
        output_dir = f'five_fold_splits/{file_type}/fold_{fold_idx}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process train files
        train_features = []
        train_labels = []
        train_participant_ids = []
        train_segment_info = []
        
        for file_info in fold_info['train_files']:
            filepath = file_info['filepath']
            label = file_info['label']
            participant_id = file_info['participant_id']
            
            if filepath in file_features:
                segments = file_features[filepath]
                train_features.extend(segments)
                train_labels.extend([label] * len(segments))
                train_participant_ids.extend([participant_id] * len(segments))
                
                # Add segment info
                for seg_idx in range(len(segments)):
                    train_segment_info.append(f"{participant_id}_seg{seg_idx}")
        
        # Process validation files
        val_features = []
        val_labels = []
        val_participant_ids = []
        val_segment_info = []
        
        for file_info in fold_info['val_files']:
            filepath = file_info['filepath']
            label = file_info['label']
            participant_id = file_info['participant_id']
            
            if filepath in file_features:
                segments = file_features[filepath]
                val_features.extend(segments)
                val_labels.extend([label] * len(segments))
                val_participant_ids.extend([participant_id] * len(segments))
                
                # Add segment info
                for seg_idx in range(len(segments)):
                    val_segment_info.append(f"{participant_id}_seg{seg_idx}")
        
        # Check if we have enough data
        if len(train_features) == 0:
            print(f"Warning: No training features for fold {fold_idx}. Skipping this fold.")
            continue
            
        if len(val_features) == 0:
            print(f"Warning: No validation features for fold {fold_idx}. Skipping this fold.")
            continue
        
        # Convert to numpy arrays
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        train_participant_ids = np.array(train_participant_ids)
        
        val_features = np.array(val_features)
        val_labels = np.array(val_labels)
        val_participant_ids = np.array(val_participant_ids)
        
        # Check feature dimensions
        if train_features.shape[-1] != 84:
            print(f"Warning: Train features have {train_features.shape[-1]} dimensions, expected 84. Skipping fold {fold_idx}.")
            continue
            
        if val_features.shape[-1] != 84:
            print(f"Warning: Val features have {val_features.shape[-1]} dimensions, expected 84. Skipping fold {fold_idx}.")
            continue
        
        # Normalize features (fit on train, transform both)
        scaler = StandardScaler()
        train_features_reshaped = train_features.reshape(-1, train_features.shape[-1])
        train_features_norm = scaler.fit_transform(train_features_reshaped).reshape(train_features.shape)
        
        val_features_reshaped = val_features.reshape(-1, val_features.shape[-1])
        val_features_norm = scaler.transform(val_features_reshaped).reshape(val_features.shape)
        
        # Handle NaN values
        train_features_norm = np.nan_to_num(train_features_norm)
        val_features_norm = np.nan_to_num(val_features_norm)
        
        # Save features
        np.save(os.path.join(output_dir, 'features_train.npy'), train_features_norm)
        np.save(os.path.join(output_dir, 'labels_train.npy'), train_labels)
        np.save(os.path.join(output_dir, 'participant_ids_train.npy'), train_participant_ids)
        np.save(os.path.join(output_dir, 'segment_info_train.npy'), train_segment_info)
        
        np.save(os.path.join(output_dir, 'features_val.npy'), val_features_norm)
        np.save(os.path.join(output_dir, 'labels_val.npy'), val_labels)
        np.save(os.path.join(output_dir, 'participant_ids_val.npy'), val_participant_ids)
        np.save(os.path.join(output_dir, 'segment_info_val.npy'), val_segment_info)
        
        # Save scaler for later use
        import pickle
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Fold {fold_idx} complete:")
        print(f"  Train: {len(train_features)} segments from {len(np.unique(train_participant_ids))} participants")
        print(f"  Val: {len(val_features)} segments from {len(np.unique(val_participant_ids))} participants")
    
    print(f"\nFeature extraction complete for all 5 folds of {file_type}!")

def main():
    parser = argparse.ArgumentParser(description='Extract features for 5-fold cross-validation (optimized)')
    parser.add_argument('--file_type', type=str, default='reading', 
                       choices=['reading', 'picture', 'free'],
                       help='Type of audio files to process')
    
    args = parser.parse_args()
    extract_features_for_all_folds(args.file_type)

if __name__ == '__main__':
    main() 