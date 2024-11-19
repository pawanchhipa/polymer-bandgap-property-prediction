import unittest
import pandas as pd
import numpy as np
import json
import os
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class TestPolymerModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.data = pd.read_csv('polymer_data.csv', index_col=0)
        
        # Load results if they exist
        if os.path.exists('results/model_performance/rf_model_metrics.json'):
            with open('results/model_performance/rf_model_metrics.json', 'r') as f:
                cls.results = json.load(f)
    
    def test_data_integrity(self):
        """Test if data meets basic requirements"""
        self.assertGreater(len(self.data), 0, "Dataset should not be empty")
        self.assertIn('smiles', self.data.columns, "SMILES column should exist")
        self.assertIn('value', self.data.columns, "Value column should exist")
        
    def test_smiles_validity(self):
        """Test if SMILES strings are valid"""
        sample = self.data['smiles'].head()
        for smile in sample:
            mol = Chem.MolFromSmiles(smile)
            self.assertIsNotNone(mol, f"Invalid SMILES string: {smile}")
            
    def test_model_performance(self):
        """Test if model performance meets minimum requirements"""
        if hasattr(self, 'results'):
            self.assertLess(self.results['mean_rmse_val'], 2.0, 
                          "RMSE should be less than 2.0")
            self.assertGreater(self.results['mean_r2_val'], 0.5,
                             "R² should be greater than 0.5")
            
    def test_results_saved(self):
        """Test if results files exist"""
        self.assertTrue(os.path.exists('results/model_performance/rf_model_metrics.json'),
                       "Results file should exist")
        self.assertTrue(os.path.exists('results/figures/parity_plot.png'),
                       "Parity plot should exist")

if __name__ == '__main__':
    unittest.main(verbosity=2)
