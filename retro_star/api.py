import logging

from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from retro_star.common import prepare_starting_molecules, \
    prepare_molstar_planner, smiles_to_fp_1, onmt_trans
from retro_star.utils.scoring_func import prepare_template_set, scoring
import numpy as np
import torch
from joblib import Parallel, delayed
import time


class RSPlanner(object):
    def __init__(self, gpu, mode, expansion_topk, iterations, use_value_fn, do_rerank, buliding_block_path, fp_dim,
                 one_step_model_path, value_fn_model_path, beam_size, route_topk, viz, viz_dir):

        setup_logger()
        self.device = torch.device('cuda:%d' % gpu if gpu >= 0 and torch.cuda.is_available() else 'cpu')
        #print(device)
        starting_mols = prepare_starting_molecules(buliding_block_path)
        beam_size = beam_size if beam_size > expansion_topk else expansion_topk
        #one_step_handler = lambda x: askcos(x, topk=expansion_topk)
        
        one_step_handler = lambda x: onmt_trans(
            x,
            topk=expansion_topk,
            model_path=one_step_model_path,
            beam_size=beam_size,
            mode=mode,
            device=gpu
        )
        
        self.top_k = route_topk
        
        
        if do_rerank:
            enzymatic_template = prepare_template_set("dataset/rxn_data/enzymatic_rule.json")
            uspto_template = prepare_template_set("dataset/rxn_data/uspto_rule.json")
            def rerank_fn(prod, result):
                reactants = result['reactants']
                
                scores_T = result['scores']
                types = result['types']
                templates = []
                scores_R = []
                scores_new = []
                ref_rxns = []
                num_refs = []

                for i in range(0, len(scores_T)):
                    if types[i] == '<B/C>':
                        data_pair = [(template, reactants[i], prod) for template in enzymatic_template]
                        data_pair.extend((template, reactants[i], prod) for template in uspto_template)
                    elif types[i] == 'chem':
                        data_pair = [(template, reactants[i], prod) for template in uspto_template]
                    else:
                        data_pair = [(template, reactants[i], prod) for template in enzymatic_template]
                    
                    res = Parallel(n_jobs=-1)(delayed(scoring)(template, react, prod) for template, react, prod in data_pair)
                    max_res = max(res, key=lambda x:x['score_R'])
                    if max_res['score_R'] == 0:
                        template = ''
                        ref_rxn = ""
                        num_ref = 0
                        score_R = 0.5
                    else:
                        template = max_res['template']
                        ref_rxn = template['ref_rxn'][max_res['ref_idx']]
                        num_ref = len(template['ref_mol'])
                        template = template['smarts']
                        score_R = max_res['score_R']
                    templates.append(template)
                    ref_rxns.append(ref_rxn)
                    num_refs.append(num_ref)
                    scores_R.append(score_R)
                    scores_new.append(scores_T[i]*(score_R + num_ref)/(1+num_ref))  
                result['scores_new'] = scores_new
                result['templates'] = templates
                result['scores_R'] = scores_R
                result['num_rxns'] = num_refs
                result['ref_rxns'] =  ref_rxns
                return result
        else:
            def rerank_fn(prod, result):
                scores_T = result['scores']
                result['scores_new'] = scores_T
                result['templates'] = ["" for _ in scores_T]
                result['scores_R'] = [0 for _ in scores_T]
                result['num_rxns'] = [0 for _ in scores_T]
                result['ref_rxns'] =  ["" for _ in scores_T]
                return result
                
        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=self.device
            ).to(self.device)
            logging.info('Loading value nn from %s' % value_fn_model_path)
            model.load_state_dict(torch.load(value_fn_model_path, map_location=self.device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp_1(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(self.device)
                v = model(fp).item()
                return v
            
        else:
            value_fn = lambda x: 0.
        self.plan_handle = prepare_molstar_planner(
            expansion_handler=one_step_handler,
            value_fn=value_fn,
            rerank_fn=rerank_fn,
            starting_mols=starting_mols,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            route_topk=route_topk
        )
    def plan(self, target_mol):
        succ, msg = self.plan_handle(target_mol)

        #if succ:
        ori_list = msg[3]
        routes_list = []
        for i in ori_list:
            routes_list.append(i.serialize_with_score())
        rxn_list = []
        for item in routes_list[:self.top_k]:
            rxns = item[0].split("|")
            for rxn in rxns:
                react = rxn.split(">")[-1]
                prod = rxn.split(">")[0]
                if react+">>"+prod not in rxn_list:
                    rxn_list.append(react+">>"+prod)
        #print(rxn_list)
        return succ, routes_list[:self.top_k]
        #return succ, routes_list, condition
            
        #else:
        #    logging.info('Synthesis path for %s not found. Please try increasing '
        #                 'the number of iterations.' % target_mol)
        #    return None


if __name__ == '__main__':
    planner = RSPlanner(
        gpu=0,
        use_value_fn=True,
        iterations=100,
        expansion_topk=50
    )

    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)

    result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    print(result)

    result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    print(result)

