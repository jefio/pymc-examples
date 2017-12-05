"""
Fit models and plot the results.
"""
import argparse
import logging
from os.path import expanduser

import numpy as np
import pymc3 as pm

from models import get_bernoulli_mixture
from twenty_tools import TwentyNewsGroups


logger = logging.getLogger(__name__)


def _get_pred_clusters(trace):
    pred_clusters = [np.argmax(np.bincount(zi)) for zi in trace.T]
    return np.array(pred_clusters, int)


class FitPlot(object):
    def __init__(self, data_params, model_params, trace_params, exp_name):
        self.trace_params = trace_params
        self.exp_name = exp_name

        self.tng = TwentyNewsGroups(data_params['max_df'], data_params['min_df'],
                                    data_params['doc_len_min'], data_params['doc_len_max'])
        self.model = get_bernoulli_mixture(self.tng.dataset['X_bin'], model_params)

    def fit_plot(self):
        with self.model:
            trace = pm.sample(self.trace_params['samples'], njobs=self.trace_params['njobs'])
        pred_clusters = _get_pred_clusters(trace['z'])
        logger.info("pred_clusters=%s", np.bincount(pred_clusters))
        filename = expanduser("~/plot/{}_clusters.png".format(self.exp_name))
        self.tng.plot_clustering(pred_clusters, filename)
        filename = expanduser("~/plot/{}_words.csv".format(self.exp_name))
        self.tng.write_words(pred_clusters, filename)


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
