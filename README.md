# ECG-AI: Deep Learning for Cardiac Hypertrophy Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based system for automatic detection of cardiac hypertrophy from 12-lead ECG signals using advanced CNN-LSTM hybrid neural networks.

## 🔬 Overview

This project implements a comprehensive machine learning pipeline for detecting cardiac hypertrophy (left ventricular hypertrophy, right ventricular hypertrophy, and atrial enlargement) from electrocardiogram (ECG) signals. The system combines convolutional neural networks (CNNs) for spatial feature extraction with long short-term memory (LSTM) networks for temporal pattern recognition.

### Key Features

- **Advanced Signal Processing**: Bandpass filtering, noise reduction, and normalization
- **Hybrid Architecture**: CNN-LSTM model for both spatial and temporal feature learning
- **Multi-task Learning**: Simultaneous hypertrophy detection and hypertrophy probability scoring
- **Clinical Decision Support**: Risk stratification with clinical recommendations
- **Class Imbalance Handling**: Weighted training for imbalanced datasets
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and clinical metrics

## 📊 Dataset

The project uses the ***[PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.1/)**, the largest publicly available ECG dataset with:

- 21,837 clinical 12-lead ECG records
- 10-second recordings at 100Hz and 500Hz sampling rates
- Comprehensive diagnostic annotations using SCP-ECG codes
- Standardized train/test splits for reproducible research

### Hypertrophy Categories Detected

- **LVH**: Left Ventricular Hypertrophy
- **RVH**: Right Ventricular Hypertrophy  
- **LAE**: Left Atrial Enlargement
- **RAE**: Right Atrial Enlargement
- **SEHYP**: Septal Hypertrophy

## 🏗️ Architecture

### Model Components

1. **Data Preprocessing Pipeline**
   - Missing data imputation: KNN and median for numerical columns, mode for categorical
   - Bandpass filtering (0.5-40 Hz) for noise removal
   - Per-sample Z-score normalization
   - Signal length standardization (1000 samples at 100Hz, 5000 at 500Hz)

2. **CNN Feature Extractor**

   ```text
   Conv1D(32, 5) → BatchNorm → MaxPool1D(2)
   Conv1D(64, 5) → BatchNorm → MaxPool1D(2)
   Conv1D(128, 3) → BatchNorm → MaxPool1D(2)
   ```

3. **LSTM Temporal Processor**

   ```text
   LSTM(128, return_sequences=True, dropout=0.3)
   LSTM(64, dropout=0.3)
   ```

4. **Classification Head**

   ```text
   Dense(128, relu) → Dropout(0.5)
   Dense(64, relu) → Dropout(0.3)
   Dense(1, sigmoid)
   ```

### Model Performance

- **Input Shape**: (1000, 12) - 1000 time points × 12 ECG leads
- **Training Strategy**: Stratified k-fold cross-validation
- **Class Weighting**: Balanced approach for imbalanced classes
- **Optimization**: Adam optimizer with learning rate scheduling

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install pandas numpy matplotlib seaborn
pip install scikit-learn wfdb scipy
pip install missingno
```

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Arushi-Gupta13/ECG-AI.git
   cd ECG-AI
   ```

2. **Download the PTB-XL dataset**

   ```python
   # The notebook includes automatic dataset download from Kaggle
   # Ensure you have kaggle.json credentials configured
   ```

3. **Run the analysis**

   ```bash
   jupyter notebook ECG_Hypertrophy.ipynb
   ```

## 📋 Project Structure

```text
ECG-AI/
├── ECG_Hypertrophy.ipynb          # Main analysis notebook
├── hypertrophy_detection_model.h5  # Trained model weights
├── preprocessing_params.pkl        # Preprocessing parameters
├── ptbxl_database_clean.csv       # Cleaned dataset
├── README.md                      # Project documentation
└── .git/                          # Git version control
```

## 🔧 Usage

### Training a New Model

```python
# Load and preprocess data
df_clean = pd.read_csv("ptbxl_database_clean.csv")
X_ecg = load_ecg_signals(df_clean)
X_processed = preprocess_ecg_signals(X_ecg)

# Prepare train/test splits using PTB-XL recommended folds
test_fold = 10
train_idx = df_clean['strat_fold'] != test_fold
test_idx = df_clean['strat_fold'] == test_fold

X_train = X_processed[train_idx]
X_test = X_processed[test_idx]
y_train = df_clean[train_idx]['has_hypertrophy'].values
y_test = df_clean[test_idx]['has_hypertrophy'].values

# Further split train into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Build and train model
model = build_cnn_lstm_model(input_shape=(1000, 12))
history = model.fit(X_train, y_train, 
                   validation_data=(X_val, y_val),
                   epochs=100, batch_size=32)
```

### Making Predictions

```python
# Load trained model
model = tf.keras.models.load_model('hypertrophy_detection_model.h5')

# Predict on new ECG signal
risk_assessment = predict_hypertrophy_risk(ecg_signal, model, preprocessing_params)
print(f"Hypertrophy Probability: {risk_assessment['hypertrophy_probability']:.3f}")
print(f"Risk Category: {risk_assessment['risk_category']}")
```

### Clinical Decision Support

The system provides three risk categories based on ECG-detected hypertrophy probability (not general cardiovascular risk):

- **High Risk (>0.8)**: Immediate echocardiography recommended
- **Moderate Risk (0.5-0.8)**: Consider echocardiography and clinical correlation  
- **Low Risk (<0.5)**: Routine follow-up appropriate

*Note: Risk categories correspond to probability of hypertrophy detected from ECG analysis, not definitive hypertension risk or future cardiovascular event prediction.*

## 📈 Results & Evaluation

### Model Performance Metrics

- **Evaluation Methods**: Classification report, confusion matrix, ROC-AUC
- **Cross-validation**: Stratified k-fold for robust performance estimation
- **Clinical Relevance**: Risk stratification aligned with cardiology guidelines

### Visualization Features

- Missing data heatmaps and patterns
- ECG signal preprocessing comparisons
- Training history and learning curves
- ROC curves and performance metrics
- Confusion matrices for error analysis

## 🔬 Technical Details

### Signal Processing

- **Sampling Rate**: 100 Hz (configurable for 500 Hz)
- **Signal Length**: 1000 samples (10 seconds)
- **Filtering**: 4th-order Butterworth bandpass filter
- **Normalization**: Per-sample, per-lead Z-score normalization

### Data Handling

- **Missing Data**: KNN imputation with distance weighting
- **Class Imbalance**: Balanced class weights during training
- **Data Splits**: PTB-XL recommended stratified splits
- **Augmentation**: Optional signal augmentation techniques

### Model Architecture Variants

1. **Primary Model**: CNN-LSTM hybrid for hypertrophy detection
2. **Risk Model**: Continuous risk score prediction
3. **Multi-task Model**: Joint hypertrophy detection and risk assessment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{ecg-ai-hypertrophy,
  title={ECG-AI: Deep Learning for Cardiac Hypertrophy Detection},
  author={Arushi Gupta},
  year={2025},
  url={https://github.com/Arushi-Gupta13/ECG-AI}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PTB-XL Dataset**: Wagner et al., "PTB-XL, a large publicly available ECG dataset"
- **TensorFlow Team**: For the deep learning framework
- **Scientific Community**: For open-source tools and methodologies

## 📞 Contact

- **Author**: Arushi Gupta, Anshika Tripathi, Kanchan Singh, Jayant Joshi
- **GitHub**: [@Arushi-Gupta13](https://github.com/Arushi-Gupta13)
- **Repository**: [ECG-AI](https://github.com/Arushi-Gupta13/ECG-AI)

---

**⚠️ Medical Disclaimer**: This software is for research purposes only and is not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.