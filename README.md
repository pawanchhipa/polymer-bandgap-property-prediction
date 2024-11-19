# Polymer Property Prediction

This project implements machine learning models to predict bandgap properties of polymers using various molecular fingerprinting techniques.

## Features
- Morgan fingerprint generation for polymers
- Multiple molecular descriptors from RDKit
- Random Forest model implementation with cross-validation
- Feature engineering and selection
- Visualization tools for model evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The main functionality is split into modules:
- `fingerprint.py`: Molecular fingerprint generation
- `model.py`: ML model implementation
- `visualization.py`: Plotting utilities
- `utils.py`: Helper functions

Run the main script:
```bash
python main.py
```

## Data
The dataset is from:
Kuenneth, Christopher, et al. "Polymer informatics with multi-task learning." Patterns 2.4 (2021).

## License
MIT License
"# polymer-bandgap-property-prediction" 
