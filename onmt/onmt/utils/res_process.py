import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
rdkit.RDLogger.logger.setLevel(4, 4)


def cano_smiles(smiles):
    if smiles == "":
        return None
    smiles = smiles.replace("[SAH]", "[32S]").replace("[CoA]", "[33S]").split(".")
    smiles_new = []
    
    for smi in smiles:
        if ("C" in smi.upper()):
            smiles_new.append(smi)
    smiles = ".".join(smiles_new)
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None
        #[a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    except:
        return None
    smi = Chem.MolToSmiles(tmp)
    #smi_flat = Chem.MolToSmiles(tmp, isomericSmiles=False)
    smi = smi.replace("[33S]", "[CoA]").replace("[32S]", "[SAH]")

    return smi
