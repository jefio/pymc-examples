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
                      alpha=params['pkw_beta_dist_alpha'],
                      beta=params['pkw_beta_dist_beta'],
                      shape=(n_comp, n_feat))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['pcomp_dirichlet_dist_alpha'] * np.ones(n_comp))
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
    n_comp = params['n_trunc']

    with pm.Model() as model:
        # sample P ~ DP(G0)
        beta = pm.Beta('beta',
                       1.,
                       params['dp_alpha'],
                       shape=n_comp)
        p_comp = pm.Deterministic(
            'p_comp',
            beta * tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]]))
        pkw = pm.Beta('pkw',
                      alpha=params['pkw_beta_dist_alpha'],
                      beta=params['pkw_beta_dist_beta'],
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
                              a=params['pcomp_dirichlet_dist_alpha'] * np.ones(n_comp))
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
                           a=params['pkw_dirichlet_dist_alpha'] * np.ones(n_feat),
                           shape=(n_comp, n_feat))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['pcomp_dirichlet_dist_alpha'] * np.ones(n_comp))
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
    n_comp = params['n_trunc']

    with pm.Model() as model:
        # sample P ~ DP(G0)
        beta = pm.Beta('beta',
                       1.,
                       params['dp_alpha'],
                       shape=n_comp)
        p_comp = pm.Deterministic(
            'p_comp',
            beta * tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]]))
        pkw = pm.Dirichlet('pkw',
                           a=params['pkw_dirichlet_dist_alpha'] * np.ones(n_feat),
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


def get_models(X_count, X_bin, params):
    bin_funcs = [
        get_beta_bernoulli_mixture,
        get_beta_bernoulli_dpmixture,
        get_logisticnormal_bernoulli_mixture
    ]
    count_funcs = [
        get_dirichlet_multinomial_mixture,
        get_dirichlet_multinomial_dpmixture
    ]
    models = {func.__name__[4:]: func(X_bin, params)
              for func in bin_funcs}
    models.update({func.__name__[4:]: func(X_count, params)
                   for func in count_funcs})
    return models


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
    return get_models(X_count, X_bin, params)
