"""
Run multiple models on the 20NG data.
"""
import argparse
import logging
import os
import json

import numpy as np
import pymc3 as pm

import models
import data


logger = logging.getLogger(__name__)
np.random.seed(123456)


def get_pred_clusters(model, samples, njobs):
    with model:
        trace = pm.sample(samples, njobs=njobs, tune=1500)
    pred_clusters = [np.argmax(np.bincount(zi)) for zi in trace['z'].T]
    return np.array(pred_clusters, int)


def save_config(config, exp_name):
    filename = os.path.expanduser("~/plot/20ng_{}_config.json".format(
        exp_name))
    with open(filename, 'w') as fwrite:
        json.dump(config, fwrite)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--max-df', type=float, default=0.3)
    parser.add_argument('--min-df', type=int, default=20)
    parser.add_argument('--doc-len-min', type=int, default=20)
    parser.add_argument('--doc-len-max', type=int, default=200)
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 1, 2, 3])
    # model
    parser.add_argument('--model-names', type=str, nargs='+')
    parser.add_argument('--n-comp', type=int, default=10)
    parser.add_argument('--n-trunc', type=int, default=30)
    parser.add_argument('--dp-alpha', type=float, default=1)
    parser.add_argument('--pcomp-dirichlet-dist-alpha', type=float, default=1)
    parser.add_argument('--pkw-beta-dist-alpha', type=float, default=1)
    parser.add_argument('--pkw-beta-dist-beta', type=float, default=1)
    parser.add_argument('--pkw-dirichlet-dist-alpha', type=float, default=1)
    # trace
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--njobs', type=int, default=1)
    # other
    parser.add_argument('--exp-name', type=str, default='A')
    args = parser.parse_args()
    logger.info("args=%s", args)

    save_config(vars(args), args.exp_name)
    dataset = data.TwentyNewsGroups(args.max_df, args.min_df, args.doc_len_min,
                                    args.doc_len_max, args.classes)
    for model_name, model in models.get_models(
            dataset.X_count, dataset.X_bin, vars(args)).items():
        if args.model_names is None or model_name in args.model_names:
            exp_name = "{}_{}".format(args.exp_name, model_name)
            pred_clusters = get_pred_clusters(model, args.samples, args.njobs)
            title= r"Clusters composition, $\alpha={}$".format(args.dp_alpha)
            dataset.evaluate_clusters(pred_clusters, exp_name, title)


if __name__ == '__main__':
    main()
