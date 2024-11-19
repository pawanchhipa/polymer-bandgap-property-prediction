import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Lipinski, GraphDescriptors

def load_polymer_data(filepath):
    """Load and preprocess polymer data"""
    polymer_data = pd.read_csv(filepath, index_col=0)
    dfs = []
    
    for p in polymer_data['property'].unique():
        df = polymer_data.loc[polymer_data['property'] == p]
        df = df[['smiles','value']]
        df.columns = ['smiles',p]
        dfs.append(df)
    
    pdb = pd.merge(dfs[2], dfs[3], on='smiles', how='outer')
    pdb = pdb.dropna(subset=['smiles']).reset_index(drop=True)
    return pdb

def generate_morgan_fingerprint(mol, n_bits=1024, radius=2):
    """Generate Morgan fingerprint for a molecule"""
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius,
        nBits=n_bits, 
        useFeatures=True
    )

def generate_molecular_descriptors(mol):
    """Generate comprehensive molecular descriptors"""
    descriptors = {}
    
    # Basic descriptors
    descriptors['MQNs'] = rdMolDescriptors.MQNs_(mol)
    descriptors['SMR_VSA'] = rdMolDescriptors.SMR_VSA_(mol)
    descriptors['SlogP_VSA'] = rdMolDescriptors.SlogP_VSA_(mol)
    
    # Additional descriptors
    descriptors['RotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    descriptors['RingCount'] = Lipinski.RingCount(mol)
    descriptors['AromaticRings'] = Lipinski.NumAromaticRings(mol)
    descriptors['AliphaticRings'] = Lipinski.NumAliphaticRings(mol)
    
    # Graph descriptors
    for desc in ['Kappa1', 'Kappa2', 'Kappa3', 
                 'Chi0', 'Chi1', 'Chi2n', 'Chi3n', 'Chi4n']:
        try:
            descriptors[desc] = getattr(GraphDescriptors, desc)(mol)
        except:
            descriptors[desc] = 0
            
    return descriptors
