# Master’s Thesis: A Deep Learning Framework for Detecting Autism Spectrum Disorder from Speech

## Abstract  
Autism Spectrum Disorder (ASD) is currently diagnosed through standardised assessments administered by clinicians who observe an individual's behaviour as they complete a series of structured tasks. This means that the diagnostic process is not only cost- and time-intensive, but also retains an element of subjectivity. To complement this traditional diagnostic pipeline, we propose a deep learning framework designed to detect ASD from speech. We first collect the data needed to train this model through Prolific by administering three standardised speech tasks to 60 people with ASD and 60 people without. To process this data, we propose a Hierarchical Attention Temporal Convolutional Network (HATCN) trained on interpretable numerical representations of speech from participants in both groups. Our results are promising. First, they reveal that our model can identify ASD with an accuracy of 83\%. Second, we find that performance differs considerably across the three tasks, indicating that the manifestation of ASD varies by context. These findings are further supported by our SHapley Additive exPlanations (SHAP) analysis, which ensures that the model remains interpretable by providing insights into speech features most relevant to the predictive process.

---

## Code Overview 

The pipeline consists of 4 main scripts that should be run in sequence:

1. **Data Preparation** - Creates the file list from audio data
2. **5-Fold Split Creation** - Creates participant-level splits for cross-validation (all audio types at once)
3. **Feature Extraction** - Extracts features for each fold based on participant splits
4. **Training & Evaluation** - Performs 5-fold cross-validation training

**Important**: This is participant-level cross-validation, meaning all segments from the same participant stay in the same fold to prevent data leakage.

## Quick Start

```bash
# 1. Prepare data (creates filelist_enhanced.csv)
python scripts/prepare_data_enhanced.py

# 2. Create all five-fold splits at once
python scripts/create_five_fold_split.py --file_type all

# 3. Extract features for each audio type
python scripts/extract_features_optimized.py --file_type reading
python scripts/extract_features_optimized.py --file_type picture
python scripts/extract_features_optimized.py --file_type free

# 4. Train and evaluate for each audio type
python scripts/train_five_fold_cv.py --file_type reading
python scripts/train_five_fold_cv.py --file_type picture
python scripts/train_five_fold_cv.py --file_type free

# 5. Run interpretability analysis
python analysis/shap_interpretability.py --file_type reading
```

## Usage

### 1. Prepare Data
```bash
python scripts/prepare_data_enhanced.py
```
This script scans the audio directories and creates a comprehensive file list (`filelist_enhanced.csv`).

### 2. Create 5-Fold Splits
```bash
# Create splits for all audio types at once (recommended)
python scripts/create_five_fold_split.py --file_type all

# Or create splits for individual audio types
python scripts/create_five_fold_split.py --file_type reading
python scripts/create_five_fold_split.py --file_type picture
python scripts/create_five_fold_split.py --file_type free
```

This creates participant-level splits ensuring all segments from the same participant stay in the same fold.

### 3. Extract Features for All Folds
```bash
python scripts/extract_features_optimized.py --file_type reading
python scripts/extract_features_optimized.py --file_type picture
python scripts/extract_features_optimized.py --file_type free
```

This extracts features for all files and distributes them across the 5 folds based on the participant splits.

### 4. Run 5-Fold Cross-Validation Training
```bash
python scripts/train_five_fold_cv.py --file_type reading
python scripts/train_five_fold_cv.py --file_type picture
python scripts/train_five_fold_cv.py --file_type free
```

Additional options:
- `--device cuda` (use GPU if available)
- `--batch_size 16` (adjust batch size)
- `--epochs 50` (maximum epochs)
- `--patience 5` (early stopping patience)

### 5. Make Predictions on New Data
```bash
# Predict on a single audio file
python scripts/predict.py --input path/to/audio.wav --file_type reading

# Predict on all audio files in a directory
python scripts/predict.py --input path/to/audio/directory --file_type reading --output results.json

# Use GPU for faster prediction
python scripts/predict.py --input path/to/audio/directory --file_type reading --device cuda
```

### 6. Interpretability Analysis

The repository includes interpretability analysis using SHAP:

#### SHAP Feature Importance Analysis
```bash
# Run SHAP analysis with gradient method (recommended)
python analysis/shap_interpretability.py --file_type reading --method gradient --n_samples 20

# Run SHAP analysis with permutation method
python analysis/shap_interpretability.py --file_type reading --method permutation --n_samples 20

# Run both methods
python analysis/shap_interpretability.py --file_type reading --method all --n_samples 20
```

**Results**: All interpretability outputs are organized in `results/interpretability_results/` folder:
- `shap/` - Feature importance plots, category analysis, and reports
  - `shap/reading/` - Results for reading audio type
  - `shap/picture/` - Results for picture audio type  
  - `shap/free/` - Results for free speech audio type

## Repository Structure

```
├── scripts/                           # Core pipeline scripts
│   ├── prepare_data_enhanced.py      # Data preparation
│   ├── create_five_fold_split.py     # Cross-validation splits
│   ├── extract_features_optimized.py # Feature extraction
│   ├── train_five_fold_cv.py         # Training and evaluation
│   └── predict.py                    # Prediction on new data
├── models/                            # Model definitions and architecture
│   └── model_interpretable.py        # HATCN model architecture
├── analysis/                          # Analysis and visualization scripts
│   ├── shap_interpretability.py      # SHAP interpretability analysis
│   ├── analyze_predictions_by_severity_corrected.py
│   ├── analyze_task_correlations.py  # Task correlation analysis
│   └── plot_confusion_matrices.py    # Confusion matrix visualization
├── results/                           # All results and outputs
│   ├── plots/                        # Generated plots and visualizations
│   ├── analysis/                     # Analysis results (JSON files)
│   ├── five_fold_splits/             # Cross-validation data and trained models
│   │   ├── reading/
│   │   ├── picture/
│   │   └── free/
│   └── interpretability_results/     # SHAP analysis results
│       └── shap/
│           ├── reading/
│           ├── picture/
│           └── free/
├── step3-normalized_data/            # Raw audio data
├── filelist_enhanced.csv             # Generated file list
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## Results

The training script provides:

1. **Individual Fold Results** - Performance metrics for each fold
2. **Aggregated Results** - Mean ± standard deviation across all folds
3. **Participant-Level Evaluation** - Performance after aggregating segment predictions to participant-level decisions

## Key Features

- **Participant Independence**: All segments from the same participant are kept together to prevent data leakage
- **Stratified Splits**: Maintains class balance across folds
- **Comprehensive Feature Extraction**: Extracts 84 features at the frame level, allowing for a comprehensive temporal assessment
- **Efficient Processing**: Option to create all splits at once or extract features once for all files
- **Overfitting Mitigation**: Multiple strategies to prevent overfitting including early stopping with patience-based stopping, dropout layers in the model architecture and regularization techniques
- **5-Fold Cross-Validation**: Robust evaluation through 5-fold cross-validation, providing more reliable and generalizable results
- **Participant-Level Metrics**: Focus on participant-level classification performance with comprehensive metrics (Accuracy, F1, Precision, Recall, TPR, TNR)
- **Reproducibility**: Fixed random seeds for consistent results
- **Clinical Interpretability**: SHAP-based feature importance analysis for clinical insights

## Model Architecture

The HATCN (Hierarchical Attention Temporal Convolutional Network) model:
- Input: 84-dimensional features from 4-second audio segments
- TCN layers with dilations [1, 2, 4, 8]
- Multi-scale region attention
- Standard attention mechanism
- Output: 2 classes (ASD vs non-ASD)
- Final prediction: Aggregated participant-level classification

## Interpretability Features

### SHAP Feature Importance Analysis
- **Feature-level importance**: Quantifies the contribution of each of the 84 features to ASD detection
- **Category-based analysis**: Groups features by type (pitch, energy, formants, MFCC, etc.) for clinical interpretation
- **Statistical significance**: Provides confidence intervals and standard deviations for feature importance
- **Clinical insights**: Identifies which speech characteristics are most indicative of ASD

## Example Output

```
5-FOLD CROSS-VALIDATION RESULTS FOR READING
================================================================================

INDIVIDUAL FOLD RESULTS:
------------------------------------------------------------

Fold 0:
  Participant Level - Acc: 0.917, F1: 0.914, Precision: 0.889, Recall: 0.941

...

AGGREGATED RESULTS (Mean ± Std)
============================================================

Participant Level:
  ACCURACY: 0.917 ± 0.023
  F1: 0.914 ± 0.028
  PRECISION: 0.889 ± 0.035
  RECALL: 0.941 ± 0.019

INTERPRETABILITY ANALYSIS RESULTS
================================================================================

Top 10 Most Important Features for READING:
  1. f0                    : 0.0456 ± 0.0123
  2. mfcc_1                : 0.0421 ± 0.0098
  3. jitter                : 0.0389 ± 0.0112
  4. energy                : 0.0356 ± 0.0089
  5. f1                    : 0.0321 ± 0.0076
  6. mfcc_delta_2          : 0.0298 ± 0.0091
  7. voicing_strength      : 0.0287 ± 0.0065
  8. mfcc_5                : 0.0264 ± 0.0083
  9. f2                    : 0.0243 ± 0.0059
 10. docc_3                : 0.0221 ± 0.0072

Feature Category Importance:
  Pitch           : 0.0456
  Voice Quality   : 0.0338
  Energy          : 0.0356
  Formants        : 0.0283
  MFCC            : 0.0312
  MFCC Delta      : 0.0256
  MFCC Delta2     : 0.0218
  DOCC            : 0.0198
  DOCC Delta      : 0.0187

Attention Analysis Summary:
  Total Samples Analyzed: 100
  ASD Samples: 52, Non-ASD Samples: 48
  Correct Predictions: 87/100 (87.0%)
  Average Confidence: 0.823
``` 
