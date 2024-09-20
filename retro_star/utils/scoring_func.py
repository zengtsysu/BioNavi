from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from retro_star.common.smiles_to_fp import smiles_to_fp, batch_smiles_to_fp
import json
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def prepare_template_set(filename):
    with open(filename, encoding="utf-8") as f:
        rxn_template = json.load(f)
    return rxn_template

def similarity_cal(product, ref_mols):
    fp_ref = batch_smiles_to_fp(ref_mols)
    fp_cur = smiles_to_fp(product)
    smilaritys = [DataStructs.TanimotoSimilarity(fp_cur, item) for item in fp_ref]      
    return smilaritys.index(max(smilaritys)), max(smilaritys)

def scoring(template, reactant, product):
    ref_idx = -1
    score_R = 0
    reactant = reactant.replace("[CoA]", "S").replace("[SAH]", "S")
    product = product.replace("[CoA]", "S").replace("[SAH]", "S")
    reactant = Chem.MolToSmiles(Chem.MolFromSmiles(reactant), isomericSmiles=False)
    patt = AllChem.ReactionFromSmarts(template['smarts'])
    mols = patt.RunReactants([Chem.MolFromSmiles(product)])
    for mol in mols:
        reactants_temp = [Chem.MolToSmiles(item, isomericSmiles=False) for item in mol]
        if set(reactant.split(".")) == set(reactants_temp): 
            ref_idx, score_R = similarity_cal(product, template['ref_mol'])
            break
    return {'template': template, 'ref_idx': ref_idx, 'score_R': score_R}
