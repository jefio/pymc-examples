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
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--max-df', type=float, default=0.5)
    parser.add_argument('--min-df', type=int, default=50)
    parser.add_argument('--doc-len-min', type=int, default=20)
    parser.add_argument('--doc-len-max', type=int, default=40)
    # model
    parser.add_argument('--n-comp', type=int, default=10)
    parser.add_argument('--beta-dist-alpha', type=float, default=2)
    parser.add_argument('--beta-dist-beta', type=float, default=100)
    parser.add_argument('--dirichlet-dist-alpha', type=float, default=1)
    # trace
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--njobs', type=int, default=1)
    # other
    parser.add_argument('--exp-name', type=str, default='A')
    args = parser.parse_args()
    logger.info("args=%s", args)

    data_params = {
        key: getattr(args, key) for key in ('max_df', 'min_df', 'doc_len_min', 'doc_len_max')
    }
    model_params = {
        'n_comp': args.n_comp,
        'beta_dist': {
            'alpha': args.beta_dist_alpha,
            'beta': args.beta_dist_beta
        },
        'dirichlet_dist': {
            'alpha': args.dirichlet_dist_alpha
        }
    }
    trace_params = {
        key: getattr(args, key) for key in ('samples', 'njobs')
    }
    exp_name = args.exp_name
    fplot = FitPlot(data_params, model_params, trace_params, exp_name)
    fplot.fit_plot()


if __name__ == '__main__':
    main()
