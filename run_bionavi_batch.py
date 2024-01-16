import os

from tqdm import tqdm
from rdkit import Chem

from config import Config
from retro_star.api import RSPlanner


def save_txt(data, path):
    with open(path, 'w') as f:
        for each in data:
            f.write(each + '\n')


def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data


def run(conf):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    # canonicalization
    mol = Chem.MolToSmiles(Chem.MolFromSmiles(conf.target_mol))

    planner = RSPlanner(
        gpu=conf.gpu,
        use_value_fn=conf.use_value_fn,
        mode=conf.mode,
        value_fn_model_path=conf.value_fn_model_path,
        fp_dim=conf.fp_dim,
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
    #print(succ)
    if result is None:
        return None

    for i, route in enumerate(result):
        route_dict = {
            'route_id': i,
            'route': route[0],
            'route_score': route[1]
        }
        result_routes_list.append(route_dict)
    return succ, result_routes_list


if __name__ == '__main__':
    conf = Config('config/bionavi_conf.yaml')
    data_path = './dataset/test_data/drugs.txt'
    data = read_txt(data_path)
    products = [each.split('\t')[0] for each in data][0:]

    for i, product in enumerate(tqdm(products)):
        
        conf.target_mol = product

        try:
            succ, result = run(conf)
        except Exception as e:
            result = None
            print(e)
        
        if result is not None:
            with open("pathway/drug_"+str(i)+".txt", "w") as f:
                for route in result:
                    f.write(route['route']+ "\t" + str(route['route_score']) + "\n")
        break
        