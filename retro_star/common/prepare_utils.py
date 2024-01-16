import os
import sys
import pickle
import logging


import pandas as pd

from retro_star.alg import molstar
from onmt.bin.translate_for_retrostar import load_model, run


def prepare_starting_molecules(filename):
    # 输入直接就是starting mols了，不从文件里读
    if isinstance(filename, list):
        logging.info('%d starting molecules loaded' % len(filename))
        return filename

    logging.info('Loading starting molecules from %s' % filename)
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename, header=None)[0]))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def onmt_trans(x, model_path, beam_size=20, topk=50, device='cpu', mode='hybrid'):
    opt, translator = load_model(
        model_path=model_path,
        beam_size=beam_size,
        topk=topk,
        mode=mode,
        device=device,
        tokenizer='token')
    res_dict = run(translator, opt, x)
    res_dict['templates'] = [None for _ in range(len(res_dict['scores']))]
    return res_dict


def prepare_molstar_planner(expansion_handler, value_fn, rerank_fn, starting_mols, iterations, viz=False, viz_dir=None, route_topk=5):
    plan_handler = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handler,
        value_fn=value_fn,
        rerank_fn=rerank_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        route_topk=route_topk
    )
    return plan_handler

