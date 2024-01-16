#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import warnings

from onmt.translate.translator import build_translator
from onmt.utils.logging import init_logger
from onmt.utils.res_process import cano_smiles
from torch.multiprocessing import Pool, set_start_method

warnings.filterwarnings("ignore")
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import numpy as np
import os
import re
import rdkit

rdkit.RDLogger.logger.setLevel(4, 4)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def translate(translator, opt, smi):
    # print("-----------opt-----------")
    # print(opt)

    if isinstance(smi, str):
        # print("---------input is a string---------")
        smi = [smi]
    # elif isinstance(smi, list):
    #    print("---------input is a list------------")
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    scores, predictions = translator.translate(
        src=smi,
        tgt=None,
        batch_size=opt.batch_size,
        batch_type=opt.batch_type,
        attn_debug=opt.attn_debug
    )
    return scores, predictions


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def add_to_score_dict(ori_dict, add_list):
    for each in add_list:
        each_key, each_val = each
        if each_key in ori_dict.keys():
            ori_dict[each_key] += each_val / 3
        else:
            ori_dict[each_key] = each_val / 3
    return ori_dict


def add_to_template_dict(ori_dict, add_list):
    for each in add_list:
        each_key, each_val = each
        ori_dict[each_key] = each_val
    return ori_dict


def aggregate(ret_list):
    if len(ret_list) > 0:
        topk = len(ret_list[0]['scores'][0])
        # print('topk: ', topk)
    score_dict_list = None
    template_dict_list = None
    for each in ret_list:
        if score_dict_list is None:
            score_dict_list = [dict() for _ in range(len(each['reactants']))]
            template_dict_list = [dict() for _ in range(len(each['reactants']))]
        for idx, each_entity in enumerate(zip(each['reactants'], each['scores'], each['templates'])):
            each_reactant, each_score, each_template = each_entity
            score_dict_list[idx] = add_to_score_dict(
                score_dict_list[idx],
                list(zip(each_reactant, each_score))
            )
            template_dict_list[idx] = add_to_template_dict(
                template_dict_list[idx],
                list(zip(each_reactant, each_template))
            )
    ret = {
        'reactants': list(),
        'scores': list(),
        'templates': list()
    }
    for idx, each_score_dict in enumerate(score_dict_list):
        ordered_tuples = sorted(
            each_score_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ordered_reactants = [each[0] for each in ordered_tuples][:topk]
        ordered_scores = [each[1] for each in ordered_tuples][:topk]
        ordered_templates = [template_dict_list[idx][each_ordered_reactants] for each_ordered_reactants in
                             ordered_reactants]

        ret['reactants'].append(ordered_reactants)
        ret['scores'].append(ordered_scores)
        ret['templates'].append(ordered_templates)

    return ret


def load_model(model_path, beam_size, topk, mode, device, tokenizer):
    parser = _get_parser()
    opt = parser.parse_args(["--model", "", "--src", ""])
    if device >= 0:
        # print("gpu mode...")
        opt.gpu = device
    else:
        # print("cpu mode...")
        opt.gpu = -1
    opt.model = model_path
    opt.beam_size = beam_size
    opt.n_best = beam_size
    opt.topk = topk
    opt.max_length = 200
    opt.tokenizer = tokenizer
    opt.mode = mode
    opt.models = opt.model
    translator = build_translator(opt, report_score=True)
    return opt, translator


def load_model_parallel(model_pth_list, gpu_list, one_step_beam_size, one_step_top_k, tokenizer):
    opt_list = []
    translator_list = []
    assert (len(gpu_list) >= len(model_pth_list))
    for i in range(len(model_pth_list)):
        opt, translator = load_model(model_pth_list[i], one_step_beam_size, one_step_top_k, gpu_list[i], tokenizer)
        opt_list.append(opt)
        translator_list.append(translator)
    return opt_list, translator_list


def translate_single(data_pkg):
    opt, translator, batch_size, x = data_pkg
    res_dict = run_batch_samples(translator, opt, x, batch_size)
    return res_dict


def run_batch_parallel(translator_list, opt_list, x, test_batch_size):
    data_pkg_list = [(opt_list[i], translator_list[i], test_batch_size, x) for i in range(len(translator_list))]
    pool = Pool(len(data_pkg_list))
    results = []
    # set_start_method('spawn')
    for res in pool.map(translate_single, data_pkg_list):
        results.append(res)
    pool.close()
    pool.join()
    return aggregate(results)


def run_batch(translator_list, opt_list, x, test_batch_size):
    data_pkg_list = [(opt_list[i], translator_list[i], test_batch_size, x) for i in range(len(translator_list))]
    results = []
    for each in data_pkg_list:
        results.append(translate_single(each))
    return aggregate(results)


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi.strip())]
    assert smi == ''.join(tokens)
    return ' '.join(tokens).strip()


def run_not_cano(translator, opt, smi):
    if opt.tokenizer == 'char':
        smi = " ".join(smi.replace(" ", ""))
    elif opt.tokenizer == "token":
        smi = smi_tokenizer(smi.replace(" ", ""))

    all_scores, all_predictions = translate(translator, opt, smi)

    all_scores = all_scores[0]
    all_predictions = all_predictions[0]
    pred = [str(each) for each in all_predictions]
    preds = [each.replace(' ', '') for each in pred]
    smiles = [item.splite("|")[0] for item in preds]
    types = [item.splite("|")[1] for item in preds]
    scores = [float(each) for each in all_scores]
    scores = list(np.exp(scores))

    topk = opt.topk
    if topk <= len(scores):
        scores = scores[0:topk]
        smiles = smiles[0:topk]
        types = types[0:topk]

    res_dict = {}
    res_dict['reactants'] = smiles
    res_dict['scores'] = scores
    res_dict['type'] = types
    # res_dict['reactants'] = preds_cano
    # res_dict['scores'] = scores_cano

    return res_dict


def run(translator, opt, smi):
    #print("translate: " + smi)
    if opt.tokenizer == 'char':
        smi = " ".join(smi.replace(" ", ""))
    elif opt.tokenizer == "token":
        smi = smi_tokenizer(smi.replace(" ", ""))

    all_scores, all_predictions = translate(translator, opt, smi)
    #print(len(all_scores[0]))
    all_scores = all_scores[0]
    all_predictions = all_predictions[0]
    pred = [str(each) for each in all_predictions]
    preds = [each.replace(' ', '') for each in pred]
    scores = [float(each) for each in all_scores]
    scores = list(np.exp(scores))
    types_cano = []
    
    smiles_cano = []
    topk_cnt = 0
    scores_tmp = []
    result_dict = {}
    for i in range(len(scores)):
        #print(preds[i])
        pred = preds[i].split("|")
        smile_cano = cano_smiles(pred[0])
        if len(pred) < 2:
            rxn_type = "B/C"
        else:
            rxn_type = pred[1]
        if smile_cano in result_dict.keys():
            result_dict[smile_cano].append(rxn_type)
        else:
            result_dict[smile_cano] = [rxn_type]
            scores_tmp.append(scores[i])
    topk_cnt = -1
    smiles_cano, types_cano, scores_cano = [], [], []
    for key, value in result_dict.items():
        topk_cnt+= 1
        if key == None:
            continue
        if set(value) == {"<B>"}:
            value = "bio"
        elif set(value) == {"<C>"}:
            value = "chem"
        else:
            value = "B/C"
        if (opt.mode == "chem") and (value == "bio"):
            continue
        if (opt.mode == "bio") and (value == "chem"):
            continue
        smiles_cano.append(key)
        types_cano.append(value)
        scores_cano.append(scores_tmp[topk_cnt])
        if len(smiles_cano) == opt.topk:
            break
    
    
    sum_scores = sum(scores_cano)
    scores_cano = [score / sum_scores for score in scores_cano]
    #scores_cano = 0.0 - np.log(np.clip(np.array(scores_cano), 0., 1.0))
    res_dict = {}
    res_dict['reactants'] = smiles_cano
    res_dict['scores'] = scores_cano
    res_dict['types'] = types_cano
    #templates = []
    res_dict['templates'] = ["" for _ in scores_cano]
    #print(res_dict)
    return res_dict


def run_batch_samples(translator, opt, smi_lst, batch_size):
    if opt.tokenizer == 'char':
        smi_lst = [" ".join(smi.replace(" ", "")) for smi in smi_lst]
    elif opt.tokenizer == "token":
        smi_lst = [smi_tokenizer(smi) for smi in smi_lst]

    opt.batch_size = batch_size

    all_scores, all_predictions = translate(translator, opt, smi_lst)

    preds = [[str(ech).replace(" ", "") for ech in ech_pred] for ech_pred in all_predictions]
    scores = [[np.exp(float(ech)) for ech in ech_score] for ech_score in all_scores]

    scores_cano = []
    preds_cano = []
    for i in range(len(scores)):
        tmp_scores_cano = []
        tmp_preds_cano = []
        topk_cnt = 0
        for j in range(len(scores[i])):
            _, pred_cano = cano_smiles(preds[i][j])
            if pred_cano == None:
                continue
            else:
                topk_cnt += 1
                tmp_scores_cano.append(scores[i][j])
                tmp_preds_cano.append(pred_cano)
            if topk_cnt >= opt.topk:
                break
        scores_cano.append(tmp_scores_cano)
        preds_cano.append(tmp_preds_cano)
    for i in range(len(scores_cano)):
        tmp_sum = sum(scores_cano[i])
        for j in range(len(scores_cano[i])):
            scores_cano[i][j] = scores_cano[i][j] / tmp_sum

    res_dict = {}
    res_dict['reactants'] = preds_cano
    res_dict['scores'] = scores_cano

    templates = []
    res_dict['templates'] = templates

    return res_dict
