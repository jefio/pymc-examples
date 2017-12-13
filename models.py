"""
Define probabilistic models.
"""
import logging

import numpy as np
import numpy.random as nr
import pymc3 as pm
import theano.tensor as tt


logger = logging.getLogger(__name__)


def get_beta_bernoulli_mixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        pkw = pm.Beta('pkw',
                      alpha=params['pkw_beta_dist']['alpha'],
                      beta=params['pkw_beta_dist']['beta'],
                      shape=(n_comp, n_feat))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['pcomp_dirichlet_dist']['alpha'] * np.ones(n_comp))
        z = pm.Categorical('z',
                           p=p_comp,
                           shape=n_doc)
        x = pm.Bernoulli('x',
                         p=pkw[z],
                         shape=(n_doc, n_feat),
                         observed=X)
    return model


def get_beta_bernoulli_dpmixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        # sample P ~ DP(G0)
        alpha = pm.Gamma('alpha',
                         1.,
                         1.)
        beta = pm.Beta('beta',
                       1.,
                       alpha,
                       shape=n_comp)
        p_comp = pm.Deterministic(
            'p_comp',
            beta * tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]]))
        pkw = pm.Beta('pkw',
                      alpha=params['pkw_beta_dist']['alpha'],
                      beta=params['pkw_beta_dist']['beta'],
                      shape=(n_comp, n_feat))
        # sample X ~ P
        z = pm.Categorical('z',
                           p=p_comp,
                           shape=n_doc)
        x = pm.Bernoulli('x',
                         p=pkw[z],
                         shape=(n_doc, n_feat),
                         observed=X)
    return model


def get_logisticnormal_bernoulli_mixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        theta = pm.MvNormal('theta',
                            mu=np.zeros(n_feat),
                            cov=np.identity(n_feat),
                            shape=(n_comp, n_feat))
        pkw = pm.Deterministic('pkw',
                               1 / (1 + tt.exp(-theta)))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['pcomp_dirichlet_dist']['alpha'] * np.ones(n_comp))
        z = pm.Categorical('z',
                           p=p_comp,
                           shape=n_doc)
        x = pm.Bernoulli('x',
                         p=pkw[z],
                         shape=(n_doc, n_feat),
                         observed=X)
    return model


def get_dirichlet_multinomial_mixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        pkw = pm.Dirichlet('pkw',
                           a=params['pkw_dirichlet_dist']['alpha'] * np.ones(n_feat),
                           shape=(n_comp, n_feat))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['pcomp_dirichlet_dist']['alpha'] * np.ones(n_comp))
        z = pm.Categorical('z',
                           p=p_comp,
                           shape=n_doc)
        x = pm.Multinomial('x',
                           n=X.sum(axis=1),
                           p=pkw[z],
                           observed=X)
    return model


def get_dirichlet_multinomial_dpmixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        # sample P ~ DP(G0)
        alpha = pm.Gamma('alpha',
                         1.,
                         1.)
        beta = pm.Beta('beta',
                       1.,
                       alpha,
                       shape=n_comp)
        p_comp = pm.Deterministic(
            'p_comp',
            beta * tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]]))
        pkw = pm.Dirichlet('pkw',
                           a=params['pkw_dirichlet_dist']['alpha'] * np.ones(n_feat),
                           shape=(n_comp, n_feat))
        # sample X ~ P
        z = pm.Categorical('z',
                           p=p_comp,
                           shape=n_doc)
        x = pm.Multinomial('x',
                           n=X.sum(axis=1),
                           p=pkw[z],
                           observed=X)
    return model


def debug():
    n_doc, n_feat, n_comp = 10, 5, 2
    X_count = nr.randint(0, 10, (n_doc, n_feat))
    X_bin = np.array(X_count > 0, int)
    params = {
        # Beta-Bernoulli
        'n_comp': n_comp,
        'pkw_beta_dist': {'alpha': 2, 'beta': 2},
        'pcomp_dirichlet_dist': {'alpha': 2},
        # Dirichlet-Multinomial
        'pkw_dirichlet_dist': {'alpha': 2}
    }
    return {
        'bbm': get_beta_bernoulli_mixture(X_bin, params),
        'bbd': get_beta_bernoulli_dpmixture(X_bin, params),
        'lbm': get_logisticnormal_bernoulli_mixture(X_bin, params),
        'dmm': get_dirichlet_multinomial_mixture(X_count, params),
        'dmd': get_dirichlet_multinomial_dpmixture(X_count, params)
    }
