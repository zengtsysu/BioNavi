import os
import numpy as np
import logging
import time

from .mol_tree import MolTree



def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn, rerank_fn, iterations, viz=False, viz_dir=None, route_topk=5):
    mol_tree = MolTree(target_mol=target_mol, known_mols=starting_mols, value_fn=value_fn, rerank_fn=rerank_fn)

    i = -1
    succ_values = []
    if not mol_tree.succ:
        for i in range(iterations):
            if not (i + 1) % 10:
                logging.info('No.%s iteration is going on ...' % (i + 1))
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            t = time.time()
            result = expand_fn(m_next.mol)
            mol_tree.expand_fn_time += (time.time() - t)
            if result is not None and (len(result['scores']) > 0):
                
                result = rerank_fn(m_next.mol, result)
                scores = result['scores_new']
                scores_T = result['scores']
                scores_R = result['scores_R']
                reactants = result['reactants']
                costs = 0.0 - np.log(np.clip(np.array(scores), 0., 1.0))
                templates = result['templates']
                types = result['types']
                ref_rxns = result['ref_rxns']
                num_rxns = result['num_rxns']
                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                mol_tree.expand(m_next, reactant_lists, costs, scores_T, scores_R, templates, num_rxns, ref_rxns, types)
                
                succ_values.append(mol_tree.root.succ_value)

            else:
                mol_tree.expand(m_next, None, None, None, None, None, None, None, None)
                succ_values.append(mol_tree.root.succ_value)
                logging.info('Expansion fails on %s!' % m_next.mol)
            
        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    routes = []
    routes = mol_tree.get_routes(iterations)
    routes = sorted(routes, key=lambda x: x.total_cost)
    possible = True
    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        f = '%s/search_tree' % (viz_dir)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, possible, i+1, routes, succ_values)
