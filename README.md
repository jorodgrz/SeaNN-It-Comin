# SeaNN It Comin' - Coastal Flood Prediction System

A deep learning system for predicting coastal flooding events using GRU (Gated Recurrent Unit) neural networks and comprehensive temporal feature engineering.

## Project Overview

This project implements an advanced machine learning system to predict coastal flooding events 1-14 days in advance across 12 coastal stations along the U.S. East Coast. The system combines physics-informed features (tidal harmonics), temporal patterns, and deep learning to achieve high-accuracy flood predictions.

## Features

- **Advanced Deep Learning**: Bidirectional GRU architecture for sequence-to-sequence flood prediction
- **Comprehensive Feature Engineering**: 74+ features including:
  - Tidal harmonic constituents (M2, S2, K1, O1, spring-neap cycles)
  - Rolling window statistics (3, 7, 14, 30 days)
  - Lag features and rate of change indicators
  - Temporal encodings (cyclical month/day representations)
  - Station-specific flood thresholds
- **Proper Temporal Validation**: Train/validation/test splits by year to prevent data leakage
- **Class Imbalance Handling**: Balanced class weights for rare flood events
- **Model Explainability**: Gradient-based feature importance analysis
- **Production-Ready**: Complete pipeline from raw data to deployment-ready models

## Performance Metrics

**Overall Test Performance (2016-2020):**
- Accuracy: 99.21%
- Precision: 77.96%
- Recall: 83.09%
- F1 Score: 80.44%
- Matthews Correlation Coefficient (MCC): 0.8008
- ROC-AUC: 99.48%

**Key Improvements vs Baseline:**
- Accuracy: +24.0% improvement
- MCC: +34.4% improvement

## Dataset

**Stations Included (12 locations):**
- Annapolis, MD
- Atlantic City, NJ
- Charleston, SC
- Eastport, ME
- Fernandina Beach, FL
- Lewes, DE
- Portland, ME
- Sandy Hook, NJ
- Sewells Point, VA
- The Battery, NY
- Washington, DC
- Wilmington, NC

**Data Coverage:**
- Training: 1950-2010 (267,360 daily samples)
- Validation: 2011-2015 (21,912 daily samples)
- Testing: 2016-2020 (21,924 daily samples)
- Source: NEUSTG (Northeastern US Tide Gauge) dataset

## Installation

### Prerequisites

- Python 3.8+
- conda (recommended for environment management)
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/jorodgrz/SeaNN-It-Comin-.git
cd SeaNN-It-Comin-

# Install dependencies
conda create -n coastal-flood python=3.9
conda activate coastal-flood
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook improved_model_comprehensive.ipynb
```

The notebook includes:
1. Data loading and preprocessing
2. Advanced feature engineering
3. Model training with early stopping
4. Comprehensive evaluation
5. Visualization generation

### Using Pre-trained Models

```python
import tensorflow as tf
import pickle
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Models/best_gru_model.keras')

# Load the feature scaler
with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your data (7 days of historical features)
# X_new should have shape: (n_samples, 7, 74)
X_new_scaled = scaler.transform(X_new.reshape(-1, 518)).reshape(-1, 7, 74)

# Generate predictions (14-day forecast)
predictions = model.predict(X_new_scaled)
# predictions shape: (n_samples, 14) - probability of flood for each of next 14 days
```

## Project Structure

```
SeaNN-It-Comin'/
├── improved_model_comprehensive.ipynb  # Main analysis notebook
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
│
├── Models/                             # Trained models
│   ├── best_gru_model.keras           # Best performing GRU model
│   ├── gru_flood_model.keras          # Final trained model
│   └── scaler.pkl                     # Feature scaler
│
├── Outputs/                            # Visualizations and results
│   ├── confusion_matrix.png           # Model confusion matrix
│   ├── gru_training_history.png       # Training curves
│   ├── lstm_feature_importance.png    # Feature importance heatmap
│   ├── per_day_metrics.png            # Performance by forecast day
│   ├── per_station_f1.png             # F1 scores by station
│   ├── top_features_lstm.png          # Top 20 important features
│   └── results.json                   # Complete results summary
│
└── Data/                               # Input data files
    ├── NEUSTG_19502020_12stations.mat # Sea level time series
    └── Seed_Coastal_Stations_Thresholds.mat # Flood thresholds
```

## Model Architecture

**GRU Neural Network:**
- Input Layer: (7 timesteps, 74 features per timestep)
- Bidirectional GRU Layer 1: 128 units with dropout (0.3)
- Bidirectional GRU Layer 2: 64 units with dropout (0.3)
- Dense Layer 1: 64 units (ReLU activation)
- Dense Layer 2: 32 units (ReLU activation)
- Output Layer: 14 units (Sigmoid activation) - one per forecast day

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Binary cross-entropy
- Batch size: 32
- Early stopping: Patience 15 epochs
- Learning rate reduction: Factor 0.5, patience 7 epochs

## Key Improvements

- Class imbalance handling with balanced class weights
- Proper temporal train/validation/test splits (no data leakage)
- Advanced feature engineering (74 features per timestep)
- Early stopping to prevent overfitting
- Comprehensive evaluation (per-day, per-station metrics)
- Model explainability with gradient-based feature importance
- Feature scaling with StandardScaler
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC)
- Production-ready code structure with modular functions

## Results and Visualizations

All visualizations and detailed results are available in the `Outputs/` directory:

- **Confusion Matrix**: Shows true positives, false positives, true negatives, false negatives
- **Training History**: Loss, accuracy, precision, and recall curves during training
- **Per-Day Metrics**: Performance degradation over 14-day forecast horizon
- **Per-Station Performance**: F1 scores vary by geographic location
- **Feature Importance**: Identifies most predictive features (e.g., days since last flood, sea level statistics)

## Future Enhancements

- Integration with real-time NOAA tide gauge data
- Web API for operational flood forecasting
- Ensemble methods combining multiple architectures
- Incorporation of weather forecast data (wind, pressure)
- Extended forecast horizons (30+ days)
- Mobile app for at-risk communities

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is available for academic and research purposes.

## Citation

If you use this code or methodology in your research, please cite:

```
Rodriguez, J. (2026). SeaNN It Comin': Deep Learning for Coastal Flood Prediction.
GitHub repository: https://github.com/jorodgrz/SeaNN-It-Comin-
```

## Acknowledgments

- NOAA for providing tide gauge data through the NEUSTG dataset
- iHARP ML Challenge for motivation and problem formulation
- TensorFlow and Keras teams for deep learning frameworks

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Last Updated**: January 2026
