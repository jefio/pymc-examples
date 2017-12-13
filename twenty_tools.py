"""
Datasets.
"""
import logging

from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


class DatasetBase(object):
    def plot_clustering(self, pred_clusters, filename):
        pxt_mat = self._get_pxt_mat(pred_clusters)
        df = pd.DataFrame(pxt_mat, columns=self.dataset['target_names'])
        df.plot.bar(stacked=True)
        plt.xlabel('Cluster ID')
        plt.ylabel('Nb of documents')
        plt.title('Clusters composition')
        plt.savefig(filename)
        plt.close()

    def _get_pxt_mat(self, pred_clusters):
        n_classes = len(self.dataset['target_names'])
        n_pred_clusters = len(set(pred_clusters))
        pxt_mat = np.zeros((n_pred_clusters, n_classes), int)
        for cdx in range(n_pred_clusters):
            idxs, = np.where(pred_clusters == cdx)
            counts = np.bincount(self.dataset['y'][idxs])
            pxt_mat[cdx][:len(counts)] = counts

        # plot clusters ordered by size
        cdxs = np.argsort(pxt_mat.sum(axis=1))[::-1]
        pxt_mat = pxt_mat[cdxs]
        logger.debug("pxt_mat=%s", pxt_mat)

        return pxt_mat


class Mnist(DatasetBase):
    def __init__(self, n_classes, feat_std_min):
        self.dataset = self._get_dataset(n_classes, feat_std_min)

    def _get_dataset(self, n_classes, feat_std_min):
        dataset = load_digits()
        n_samples = len(dataset.images)
        X_count = dataset.images.reshape((n_samples, -1))

        keep_samples = np.where(dataset.target < n_classes)
        logger.info("keep_samples=%s/%s", len(keep_samples[0]), len(X_count))
        X_count = X_count[keep_samples]
        y = dataset.target[keep_samples]

        keep_feat = np.where(X_count.std(axis=0) > feat_std_min)
        logger.info("keep_feat=%s/%s", len(keep_feat[0]), len(X_count.T))
        X_count = X_count.T[keep_feat].T

        logger.info("X=%s, mean=%s", X_count.shape, X_count.mean())
        logger.info("y=%s", np.bincount(y))
        return {
            'X_count': X_count,
            'X_bin': np.array(X_count > 0, int),
            'y': y,
            'target_names': [str(c) for c in range(n_classes)]
        }


class TwentyNewsGroups(object):
    def __init__(self, max_df, min_df, doc_len_min, dlen_max):
        self.dataset = _get_dataset(max_df, min_df, doc_len_min, dlen_max)

    def plot_clustering(self, pred_clusters, filename):
        pxt_mat = self._get_pxt_mat(pred_clusters)
        df = pd.DataFrame(pxt_mat, columns=self.dataset['target_names'])
        df.plot.bar(stacked=True)
        plt.xlabel('Cluster ID')
        plt.ylabel('Nb of documents')
        plt.title('Clusters composition')
        plt.savefig(filename)
        plt.close()

    def _get_pxt_mat(self, pred_clusters):
        n_classes = len(self.dataset['target_names'])
        n_pred_clusters = len(set(pred_clusters))
        pxt_mat = np.zeros((n_pred_clusters, n_classes), int)
        for cdx in range(n_pred_clusters):
            idxs, = np.where(pred_clusters == cdx)
            counts = np.bincount(self.dataset['y'][idxs])
            pxt_mat[cdx][:len(counts)] = counts

        # plot clusters ordered by size
        cdxs = np.argsort(pxt_mat.sum(axis=1))[::-1]
        pxt_mat = pxt_mat[cdxs]
        logger.debug("pxt_mat=%s", pxt_mat)

        return pxt_mat

    def _get_words(self, pred_clusters):
        n_pred_clusters = len(set(pred_clusters))
        pred_clusters = np.arange(len(pred_clusters)) % n_pred_clusters
        for cdx in range(n_pred_clusters):
            idxs, = np.where(pred_clusters == cdx)
            xis = self.dataset['X_tfidf'][idxs]
            fdxs = np.argsort(xis.max(axis=0))[-20:]
            top_terms = [self.dataset['terms'][fdx] for fdx in fdxs]
            yield top_terms

    def write_words(self, pred_clusters, filename):
        with open(filename, 'w') as fwrite:
            for top_terms in self._get_words(pred_clusters):
                line = ','.join(top_terms)
                fwrite.write(line + '\n')


def _get_dataset(max_df, min_df, doc_len_min, doc_len_max):
    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 remove=('headers', 'footers', 'quotes'))

    # filter terms
    tvec = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X_tfidf = tvec.fit_transform(dataset.data).toarray()
    terms = tvec.get_feature_names()
    logger.debug("terms=%s", terms)

    X_bin = np.array(X_tfidf > 0, int)
    doc_lengths = X_bin.sum(axis=1)
    keep = (doc_lengths >= doc_len_min) & (doc_lengths <= doc_len_max)
    record = {
        'X_bin': X_bin[keep],
        'X_tfidf': X_tfidf[keep],
        'y': dataset.target[keep],
        'target_names': dataset.target_names,
        'terms': terms
    }
    logger.info("X=%s", record['X_bin'].shape)
    logger.info("y=%s", np.bincount(record['y']))
    return record
