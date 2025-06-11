import numpy as np
import time
from queue import Queue
import logging
import networkx as nx
from copy import deepcopy
from graphviz import Digraph
from networkx.drawing.nx_pydot import write_dot
from .mol_node import MolNode
from .reaction_node import ReactionNode
from .syn_route import SynRoute

from rdkit import Chem


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, rerank_fn, zero_known_value=True):
        self.target_mol = target_mol
        self.value_fn = value_fn
        self.rerank_fn = rerank_fn
        self.zero_known_value = zero_known_value
        self.mol_nodes = []
        self.reaction_nodes = []
        self.value_fn_time = 0
        self.expand_fn_time = 0
        if target_mol in known_mols:
            logging.info('Target in starting molecules, remove it from starting molecules!')
            known_mols.remove(target_mol)
        mol_t = Chem.MolFromSmiles(target_mol.replace("[CoA]", "S").replace("[SAH]", "S"))
        carbons = [atom for atom in mol_t.GetAtoms() if atom.GetSymbol() == "C"]
        if len(carbons) < 4:
            self.known_mols = []
        else:
            self.known_mols = known_mols
        self.succ = target_mol in known_mols
        self.search_status = 0
        #self.known_mols = known_mols
        self.root = self._add_mol_node(target_mol, None)
    def _add_mol_node(self, mol, parent):
        
        # mol = cano_smi(mol)
        if len(self.known_mols) == 0:
            is_known = False
        else:
            mol_t = Chem.MolFromSmiles(mol.replace("[CoA]", "S").replace("[SAH]", "S"))
            carbons = [atom for atom in mol_t.GetAtoms() if atom.GetSymbol() == "C"]
            is_known = (mol in self.known_mols) or (len(carbons) < 4)
            #is_known = mol in self.known_mols

        t = time.time()
        init_value = self.value_fn(mol)
        self.value_fn_time += (time.time() - t)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        # if not is_known:
        #     mol_node.value = - mol_node.depth

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, score_T, score_R, mols, parent, template, num_rxn, ref_rxn, type, ancestors):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:
                return

        reaction_node = ReactionNode(parent, cost, score_T, score_R, template, num_rxn, ref_rxn, type)
        for mol in mols:
            self._add_mol_node(mol, reaction_node)
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, scores_T, scores_R, templates, num_rxns, ref_rxns, types):
        assert not mol_node.is_known and not mol_node.children

        if costs is None:      # No expansion results
            assert mol_node.init_values(no_child=True)[0] == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, np.inf, from_mol=mol_node.mol)
            return self.succ

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(costs[i], scores_T[i], scores_R[i], reactant_lists[i],
                                             mol_node, templates[i], num_rxns[i], ref_rxns[i], types[i], ancestors)

        assert mol_node.open
        if len(mol_node.children) == 0:      # No valid expansion results
            assert mol_node.init_values(no_child=True)[0] == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, np.inf, from_mol=mol_node.mol)
            return self.succ

        v_delta, v_plan_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, v_plan_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            logging.info('Synthesis route found!')
            self.succ = True
        
        return self.succ

    def get_routes(self,iterations):
        #if not self.succ:
        #    return None

        routes = []
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put((syn_route, [self.root]))
        
        while not mol_queue.empty():
            syn_route, mol = mol_queue.get()
            
            
            temp_mol = mol[0]
            
            if temp_mol.is_known:
                syn_route.set_value(temp_mol.mol, temp_mol.succ_value)
                
            all_reactions = []
            for reaction in temp_mol.children:
                if (self.succ) and (iterations>1):
                    if reaction.succ:
                        all_reactions.append(reaction)
                else:
                    all_reactions.append(reaction)
            
            syn_route_template = None
            mol_template = None
            
            
            if len(all_reactions) > 0:
                syn_route_template = deepcopy(syn_route)
                mol_template = deepcopy(mol[1:])
                for reaction in all_reactions:
                    
                    syn_route = deepcopy(syn_route_template)
                    mol = deepcopy(mol_template)
                    mol.extend(reaction.children)
                    reactants = []
                    end = True
                    mol_queue.put((syn_route, mol))
                    for reactant in mol:
                        if len(reactant.children) > 0:
                            end = False
                        if reactant in reaction.children:
                            reactants.append(reactant.mol)
                    syn_route.add_reaction(
                        mol=temp_mol.mol,
                        value=temp_mol.succ_value,   # might be incorrect
                        template=reaction.template,
                        type=reaction.type,
                        reactants=reactants,
                        cost=[reaction.cost, reaction.score_T, reaction.score_R],
                        num_rxns = reaction.num_rxn,
                        ref_rxns = reaction.ref_rxn
                    )
                    
                   
                    if end:  
                        routes.append(syn_route)
            else:
                if len(mol) == 1:
                    continue
                syn_route_template = deepcopy(syn_route)
                mol_template = deepcopy(mol[1:])
                mol_queue.put((syn_route_template, mol_template))

        return routes

    def viz_search_tree(self, viz_file, topk=3):
        # pass
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'svg'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.2f' % np.exp(-node.cost) + node.type
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()

    def viz_search_progress(self, route, viz_file):
        target = route[0].split('>')[0]
        assert target == self.target_mol

        reaction_dict = {}

        G = nx.DiGraph()

        for reaction in route:
            parent = reaction.split('>')[0]
            reactants = reaction.split('>')[2].split('.')
            reaction_dict[parent] = set(reactants)

            if parent not in list(G.nodes):
                G.add_node(parent)

            for reactant in reactants:
                G.add_node(reactant)
                G.add_edge(parent, reactant)

        # match
        mapping = {}
        unable_to_find = False
        match_queue = Queue()
        match_queue.put(self.root)
        while not match_queue.empty():
            node = match_queue.get()
            if node.mol not in reaction_dict:
                # starting molecule
                mapping[node.mol] = '%s | %f | CLOSE' % (node.mol, node.v_target())
                continue
            route_reactants = reaction_dict[node.mol]

            if node.open:
                mapping[node.mol] = '%s | %f | OPEN' % (node.mol, node.v_target())
                continue

            mapping[node.mol] = '%s | %f | CLOSE' % (node.mol, node.v_target())

            found_match = False
            for c in node.children:
                reactants_c = set()
                for mol_node in c.children:
                    reactants_c.add(mol_node.mol)

                if reactants_c.issubset(route_reactants):
                    for mol_node in c.children:
                        match_queue.put(mol_node)
                    found_match = True
                    continue
            if not found_match:
                unable_to_find = True

        G = nx.relabel_nodes(G, mapping, copy=False)
        G.graph['node'] = {'shape': 'rect'}
        A = nx.nx_agraph.to_agraph(G)
        A.layout('dot')
        A.draw(viz_file)

        return unable_to_find
