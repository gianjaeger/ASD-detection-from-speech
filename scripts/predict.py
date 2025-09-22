import os
import numpy as np
import torch
import librosa
import parselmouth
import argparse
import json
import pickle
from sklearn.preprocessing import StandardScaler
from model import HATCN
from tqdm import tqdm

# Feature extraction parameters (same as training)
SAMPLE_RATE = 16000
FRAME_LENGTH = int(0.03 * SAMPLE_RATE)  # 30 ms
HOP_LENGTH = int(0.015 * SAMPLE_RATE)   # 15 ms
N_MFCC = 13
N_DOCC = 13

# Segmentation parameters
SEGMENT_DURATION = 4.0
SEGMENT_OVERLAP = 1.0
SEGMENT_LENGTH = int(SEGMENT_DURATION * SAMPLE_RATE)
SEGMENT_HOP = int((SEGMENT_DURATION - SEGMENT_OVERLAP) * SAMPLE_RATE)

def extract_librosa_features(y, sr):
    """Extract librosa features (same as training)"""
    energy = np.log(np.sum(librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)**2, axis=0) + 1e-6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    docc = librosa.feature.delta(mfcc, width=5, order=1, mode='mirror')
    docc_delta = librosa.feature.delta(mfcc, width=5, order=2, mode='mirror')
    return energy, mfcc, mfcc_delta, mfcc_delta2, docc, docc_delta

def extract_parselmouth_features(y, sr):
    """Extract parselmouth features (same as training)"""
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
    """Extract features for a single audio segment (same as training)"""
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
    """Segment audio (same as training)"""
    segments = []
    start = 0
    while start + segment_length <= len(y):
        segment = y[start:start + segment_length]
        segments.append(segment)
        start += segment_hop
    return segments

def extract_features_for_file(filepath):
    """Extract features for a single audio file (same as training)"""
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

def load_model_and_scaler(file_type, device='cpu'):
    """Load the best model and scaler for a given file type"""
    # Load best model info
    best_model_info_file = f'five_fold_splits/{file_type}/best_model_info.json'
    if not os.path.exists(best_model_info_file):
        print(f"Error: {best_model_info_file} not found. Run training first.")
        return None, None
    
    with open(best_model_info_file, 'r') as f:
        best_model_info = json.load(f)
    
    # Load the best model
    model_path = best_model_info['best_model_path']
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None, None
    
    # Initialize model
    model = HATCN(input_dim=84, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load scaler from the best fold
    best_fold = best_model_info['best_fold']
    scaler_path = f'five_fold_splits/{file_type}/fold_{best_fold}/scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found.")
        return None, None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Loaded best model from fold {best_fold} with F1 score: {best_model_info['best_f1_score']:.3f}")
    return model, scaler

def predict_single_file(filepath, model, scaler, device='cpu'):
    """Make predictions for a single audio file"""
    # Extract features
    segment_features = extract_features_for_file(filepath)
    if segment_features is None:
        return None, None
    
    # Normalize features
    segment_features_reshaped = segment_features.reshape(-1, segment_features.shape[-1])
    segment_features_norm = scaler.transform(segment_features_reshaped).reshape(segment_features.shape)
    segment_features_norm = np.nan_to_num(segment_features_norm)
    
    # Convert to tensor
    X = torch.FloatTensor(segment_features_norm).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1).cpu().numpy()
        confidence_scores = probabilities.max(dim=1)[0].cpu().numpy()
    
    return predictions, confidence_scores

def predict_directory(input_dir, file_type, output_file=None, device='cpu'):
    """Make predictions for all audio files in a directory"""
    print(f"Loading model for {file_type}...")
    model, scaler = load_model_and_scaler(file_type, device)
    if model is None or scaler is None:
        return
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    results = []
    for filepath in tqdm(audio_files, desc="Processing files"):
        predictions, confidence_scores = predict_single_file(filepath, model, scaler, device)
        
        if predictions is not None:
            # Aggregate segment predictions to file-level prediction
            file_prediction = 1 if np.mean(predictions) > 0.5 else 0
            file_confidence = np.mean(confidence_scores)
            
            results.append({
                'filepath': filepath,
                'prediction': file_prediction,
                'confidence': file_confidence,
                'n_segments': len(predictions),
                'segment_predictions': predictions.tolist(),
                'segment_confidences': confidence_scores.tolist()
            })
        else:
            results.append({
                'filepath': filepath,
                'prediction': None,
                'confidence': None,
                'n_segments': 0,
                'segment_predictions': [],
                'segment_confidences': [],
                'error': 'Failed to extract features'
            })
    
    # Save results
    if output_file is None:
        output_file = f'predictions_{file_type}_{os.path.basename(input_dir)}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful_predictions = [r for r in results if r['prediction'] is not None]
    if successful_predictions:
        asd_count = sum(1 for r in successful_predictions if r['prediction'] == 1)
        non_asd_count = sum(1 for r in successful_predictions if r['prediction'] == 0)
        avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
        
        print(f"\nPrediction Summary:")
        print(f"  Total files processed: {len(results)}")
        print(f"  Successful predictions: {len(successful_predictions)}")
        print(f"  Predicted ASD: {asd_count}")
        print(f"  Predicted Non-ASD: {non_asd_count}")
        print(f"  Average confidence: {avg_confidence:.3f}")
    
    print(f"\nResults saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file or directory')
    parser.add_argument('--file_type', type=str, required=True,
                       choices=['reading', 'picture', 'free'],
                       help='Type of audio files (must match training)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file prediction
        print(f"Loading model for {args.file_type}...")
        model, scaler = load_model_and_scaler(args.file_type, device)
        if model is None or scaler is None:
            return
        
        predictions, confidence_scores = predict_single_file(args.input, model, scaler, device)
        
        if predictions is not None:
            file_prediction = 1 if np.mean(predictions) > 0.5 else 0
            file_confidence = np.mean(confidence_scores)
            
            print(f"\nPrediction Results for {args.input}:")
            print(f"  Prediction: {'ASD' if file_prediction == 1 else 'Non-ASD'}")
            print(f"  Confidence: {file_confidence:.3f}")
            print(f"  Number of segments: {len(predictions)}")
            print(f"  Segment predictions: {predictions.tolist()}")
        else:
            print(f"Failed to process {args.input}")
    
    elif os.path.isdir(args.input):
        # Directory prediction
        predict_directory(args.input, args.file_type, args.output, device)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == '__main__':
    main() 