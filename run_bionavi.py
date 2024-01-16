import os

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from config import Config
from retro_star.api import RSPlanner



def run(conf):

    # canonicalization
    coa = Chem.MolFromSmarts("*CCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(*)OP(=O)(*)OC*3O*(n2cnc1c(ncnc12)N)*(O)*3OP(=O)(*)*")
    sah = Chem.MolFromSmarts("*(**3(O*([#7]1(*2(*([#7]*1)*(*)[#7]*[#7]2)))*(*)*(*)3))CC[C@@H](C(=O)O)N")
    repl_coa = Chem.MolFromSmiles("[32S]")
    repl_sah = Chem.MolFromSmiles("[33S]")

    target_mol = conf.target_mol

    mol = Chem.MolFromSmiles(target_mol)
    while mol.HasSubstructMatch(coa):
        rms = AllChem.ReplaceSubstructs(mol,coa,repl_coa)
        mol = rms[0]
        Chem.SanitizeMol(mol)
    while mol.HasSubstructMatch(sah):
        rms = AllChem.ReplaceSubstructs(mol,sah,repl_sah)
        mol = rms[0]
        Chem.SanitizeMol(mol)

    mol = Chem.MolToSmiles(mol).replace("[32S]", "[CoA]").replace("[33S]", "[SAH]")

    
    planner = RSPlanner(
        gpu=conf.gpu,
        use_value_fn=conf.use_value_fn,
        value_fn_model_path=conf.value_fn_model_path,
        fp_dim=conf.fp_dim,
        mode=conf.mode,
        iterations=conf.iterations,
        expansion_topk=conf.expansion_topk,
        route_topk=conf.route_topk,
        buliding_block_path=conf.buliding_block_path,
        one_step_model_path=conf.one_step_model_path,
        beam_size=conf.beam_size,
        viz=conf.viz,
        viz_dir=conf.viz_dir,
        do_rerank = conf.do_rerank
    )

    succ, result = planner.plan(mol)
    result_routes_list = []

    if result is None:
        return None

    for i, route in enumerate(result):
        route_dict = {
            'route_id': i,
            'route': route[0],
            'route_score': route[1]
        }
        result_routes_list.append(route_dict)
    return result_routes_list


if __name__ == '__main__':
    conf = Config('config/bionavi_conf.yaml')
    if not os.path.exists(conf.viz_dir):
        os.makedirs(conf.viz_dir)

    try:
        result = run(conf)
    except Exception as e:
        result = None
        print(e)

    if result is not None:
        for route in result:
            print(f"route id: {route['route_id']}\n"
                    f"route score:  {route['route_score']}\n"
                    f"route: {route['route']}\n")
