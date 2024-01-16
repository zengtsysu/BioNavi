import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fp(s):
    s = s.replace("[SAH]", "S").replace("[CoA]", "S")
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprint(mol, 2)
    
    return fp

def batch_smiles_to_fp(s_list):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s))
    return np.array(fps)

def smiles_to_fp_1(s, fp_dim=2048, pack=False):
    s = s.replace("[SAH]", "*").replace("[CoA]", "*")
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)

    return arr