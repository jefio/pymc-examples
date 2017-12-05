"""
Define probabilistic models.
"""
import logging

import numpy as np
import pymc3 as pm


logger = logging.getLogger(__name__)


def get_bernoulli_mixture(X, params):
    n_doc, n_feat = X.shape
    n_comp = params['n_comp']

    with pm.Model() as model:
        pkw = pm.Beta('pkw', alpha=params['beta_dist']['alpha'],
                      beta=params['beta_dist']['beta'], shape=(n_comp, n_feat))
        p_comp = pm.Dirichlet('p_comp',
                              a=params['dirichlet_dist']['alpha'] * np.ones(n_comp))
        z = pm.Categorical('z', p=p_comp, shape=n_doc)
        x = pm.Bernoulli('x', p=pkw[z], shape=(n_doc, n_feat), observed=X)

    return model
