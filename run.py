"""
Run multiple models on the Mnist data.
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


def get_pred_clusters(model, samples, njobs):
    with model:
        trace = pm.sample(samples, njobs=njobs)
    pred_clusters = [np.argmax(np.bincount(zi)) for zi in trace['z'].T]
    return np.array(pred_clusters, int)


def save_config(config, exp_name):
    filename = os.path.expanduser("~/plot/mnist_{}_config.json".format(
        exp_name))
    with open(filename, 'w') as fwrite:
        json.dump(config, fwrite)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--n-classes', type=int, default=10)
    parser.add_argument('--feat-std-min', type=float, default=0.1)
    # model
    parser.add_argument('--model-names', type=str, nargs='+')
    parser.add_argument('--n-comp', type=int, default=10)
    parser.add_argument('--n-trunc', type=int, default=30)
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
    dataset = data.Mnist(args.n_classes, args.feat_std_min)
    model_params = {
        'n_comp': args.n_comp,
        'pcomp_dirichlet_dist': {
            'alpha': args.pcomp_dirichlet_dist_alpha
        },
        'pkw_beta_dist': {
            'alpha': args.pkw_beta_dist_alpha,
            'beta': args.pkw_beta_dist_beta
        },
        'pkw_dirichlet_dist': {
            'alpha': args.pkw_dirichlet_dist_alpha
        }
    }
    for model_name, model in models.get_models(
            dataset.X_count, dataset.X_bin, model_params).items():
        exp_name = "{}_{}".format(args.exp_name, model_name)
        pred_clusters = get_pred_clusters(model, args.samples, args.njobs)
        dataset.evaluate_clusters(pred_clusters, exp_name)


if __name__ == '__main__':
    main()
