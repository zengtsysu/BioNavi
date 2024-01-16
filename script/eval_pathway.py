import os
from rdkit import Chem


gt_route = {}
n = 0
for line in open("./dataset/test_data/biosynthetic_set.txt"):
    data = line.strip().split("\t")
    route = []
    
    blocks = []
    branches = 0
    mols = []
    rxns = data[1].split("|")

    for rxn in rxns:
        src = rxn.split(">>")[0]
        tar = rxn.split(">>")[1]
        route.append(tar)
    if n in gt_route.keys():
        gt_route[n].append((route, blocks))
    else:
        gt_route[n] = [(route, blocks)]
    n += 1
with open("result.txt", "w") as f:
    f.write("mol_id\tsuccess\tpathway_hit\tsolution\n")
for mol_id in gt_route.keys():
    match = False
    solution = 0
    succeed = False
    match_list = []
    max_id = 1
    if os.path.exists("pathway/mol_"+str(mol_id)+".txt"):
        succeed = True
        for line1 in open("pathway/mol_"+str(mol_id)+".txt"):
            if ">" not in line1:
                continue
            solution += 1
            pred_route = []
            data1 = line1.strip().split("\t")
            rxns = data1[0].split("|")
            pred_blocks = []
            for rxn in rxns:
                rxn = rxn.replace("[CoA]", "[33S]").replace("[SAH]", "[32S]")
                src = rxn.split(">")[0]
                tar = rxn.split(">")[2]
                tar = Chem.MolToSmiles(Chem.MolFromSmiles(tar))
                pred_route.append(tar.replace("[33S]", "[CoA]").replace("[32S]", "[SAH]"))
            for pair in gt_route[mol_id]:
                intersect = len(set(pair[0]) & set(pred_route))
                if set(pair[0]) == set(pred_route):
                    match = True
    with open("result.txt", "a") as f:
        f.write("mol_"+str(mol_id) + "\t" + str(succeed) + "\t" + str(match) + "\t" + str(solution) + "\n")
