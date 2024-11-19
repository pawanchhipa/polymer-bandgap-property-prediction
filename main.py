import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils import load_polymer_data, generate_morgan_fingerprint, generate_molecular_descriptors
from visualization import plot_parity, plot_feature_importance
from model import PolymerPropertyPredictor

def main():
    # Load data
    print("Loading polymer data...")
    pdb = load_polymer_data('polymer_data.csv')
    
    # Generate features
    print("Generating molecular features...")
    morgan_fps = []
    mol_descs = []
    
    for smile in tqdm(pdb['smiles'].values):
        mol = Chem.MolFromSmiles(smile)
        
        # Morgan fingerprint
        fp = generate_morgan_fingerprint(mol)
        morgan_fps.append([int(b) for b in fp.ToBitString()])
        
        # Molecular descriptors
        descs = generate_molecular_descriptors(mol)
        mol_descs.append(descs)
    
    # Create feature matrices
    X_morgan = pd.DataFrame(morgan_fps, columns=[f'fp_morgan_{i}' for i in range(len(morgan_fps[0]))])
    X_desc = pd.DataFrame(mol_descs)
    
    # Combine features
    X = pd.concat([X_morgan, X_desc], axis=1)
    
    # Prepare target variable (Egc)
    data = pd.concat([pdb, X], axis=1)
    egc_data = data.dropna(subset=['Egc']).sample(frac=1, random_state=42)
    
    X = egc_data[X.columns]
    y = egc_data['Egc']
    
    # Initialize and train model
    print("Training model...")
    model = PolymerPropertyPredictor()
    
    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = model.optimize_hyperparameters(X, y)
    print(f"Best parameters: {best_params}")
    
    # Select features
    print("Selecting features...")
    selected_features = model.select_features(X, y)
    X = X[selected_features]
    
    # Cross-validate
    print("Performing cross-validation...")
    cv_results = model.cross_validate(X, y)
    print("\nCross-validation results:")
    print(cv_results.mean())
    
    # Plot feature importance
    plot_feature_importance(model.model, X.columns)
    plt.show()

if __name__ == "__main__":
    main()