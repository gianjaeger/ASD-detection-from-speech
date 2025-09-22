import os
import pandas as pd
import argparse

def scan_audio_directories(base_dir="step3-normalized_data"):
    """
    Scan audio directories and create a comprehensive file list.
    
    Args:
        base_dir (str): Base directory containing ASD and non-ASD speech segments
        
    Returns:
        list: List of dictionaries containing file information
    """
    file_list = []
    
    # Define the expected structure
    asd_dir = os.path.join(base_dir, "ASD speech segments")
    non_asd_dir = os.path.join(base_dir, "non-ASD Speech segments ")  # Note the trailing space
    3
    # Process ASD files
    if os.path.exists(asd_dir):
        print(f"Scanning ASD directory: {asd_dir}")
        for participant_dir in os.listdir(asd_dir):
            participant_path = os.path.join(asd_dir, participant_dir)
            if os.path.isdir(participant_path):
                for filename in os.listdir(participant_path):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(participant_path, filename)
                        
                        # Extract information from filename
                        # Format: participant_id_snippetType_timestamp.wav
                        parts = filename.replace('.wav', '').split('_')
                        participant_id = parts[0]
                        
                        # Determine snippet type
                        if 'freeSpeech' in filename:
                            snippet_type = 'free'
                        elif 'pictureDescription' in filename:
                            snippet_type = 'picture'
                        elif 'reading' in filename:
                            snippet_type = 'reading'
                        else:
                            print(f"Warning: Unknown snippet type in {filename}")
                            continue
                        
                        file_list.append({
                            'filepath': filepath,
                            'participant_id': participant_id,
                            'snippet_type': snippet_type,
                            'label': 1,  # ASD = 1
                            'filename': filename
                        })
    
    # Process non-ASD files
    if os.path.exists(non_asd_dir):
        print(f"Scanning non-ASD directory: {non_asd_dir}")
        for participant_dir in os.listdir(non_asd_dir):
            participant_path = os.path.join(non_asd_dir, participant_dir)
            if os.path.isdir(participant_path):
                for filename in os.listdir(participant_path):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(participant_path, filename)
                        
                        # Extract information from filename
                        parts = filename.replace('.wav', '').split('_')
                        participant_id = parts[0]
                        
                        # Determine snippet type
                        if 'freeSpeech' in filename:
                            snippet_type = 'free'
                        elif 'pictureDescription' in filename:
                            snippet_type = 'picture'
                        elif 'reading' in filename:
                            snippet_type = 'reading'
                        else:
                            print(f"Warning: Unknown snippet type in {filename}")
                            continue
                        
                        file_list.append({
                            'filepath': filepath,
                            'participant_id': participant_id,
                            'snippet_type': snippet_type,
                            'label': 0,  # non-ASD = 0
                            'filename': filename
                        })
    
    return file_list

def main():
    parser = argparse.ArgumentParser(description='Prepare data by scanning audio directories and creating file list')
    parser.add_argument('--output', type=str, default='filelist_enhanced.csv', 
                       help='Output CSV file path')
    parser.add_argument('--base_dir', type=str, default='step3-normalized_data',
                       help='Base directory containing audio data')
    
    args = parser.parse_args()
    
    print("Scanning audio directories...")
    file_list = scan_audio_directories(args.base_dir)
    
    if not file_list:
        print("Error: No audio files found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(file_list)
    
    # Sort by participant_id and snippet_type for consistency
    df = df.sort_values(['participant_id', 'snippet_type', 'filename']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    # Print summary
    print(f"\nFile list created: {args.output}")
    print(f"Total files: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"ASD files: {len(df[df['label'] == 1])}")
    print(f"Non-ASD files: {len(df[df['label'] == 0])}")
    print(f"\nFiles per snippet type:")
    for snippet_type in ['reading', 'picture', 'free']:
        count = len(df[df['snippet_type'] == snippet_type])
        print(f"  {snippet_type}: {count}")

if __name__ == "__main__":
    main() 