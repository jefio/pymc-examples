"""
Datasets.
"""
import logging
import os
from collections import Counter

from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.porter import PorterStemmer
from scipy.stats import describe


logger = logging.getLogger(__name__)


class DatasetBase(object):
    def plot_clustering(self, pred_clusters, filename, title):
        pxt_mat = self._get_pxt_mat(pred_clusters)
        df = pd.DataFrame(pxt_mat, columns=self.target_names)
        df.plot.bar(stacked=True)
        plt.xlabel('Cluster ID')
        plt.ylabel('Nb of documents')
        plt.title(title)
        plt.savefig(filename)
        plt.close()

    def _get_pxt_mat(self, pred_clusters):
        n_classes = len(self.target_names)
        cluster_ids = set(pred_clusters)
        pxt_mat = np.zeros((len(cluster_ids), n_classes), int)
        for cdx, cluster_id in enumerate(cluster_ids):
            idxs, = np.where(pred_clusters == cluster_id)
            counts = np.bincount(self.y[idxs])
            pxt_mat[cdx][:len(counts)] = counts

        # plot clusters ordered by size
        cdxs = np.argsort(pxt_mat.sum(axis=1))[::-1]
        pxt_mat = pxt_mat[cdxs]
        logger.debug("pxt_mat=%s", pxt_mat)
        assert pxt_mat.sum() == len(pred_clusters)
        return pxt_mat


class Mnist(DatasetBase):
    def __init__(self, n_classes, feat_std_min):
        dataset = self._get_dataset(n_classes, feat_std_min)
        self.X_count = dataset['X_count']
        self.X_bin = dataset['X_bin']
        self.y = dataset['y']
        self.target_names = dataset['target_names']

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

    def evaluate_clusters(self, pred_clusters, exp_name):
        filename = os.path.expanduser("~/plot/mnist_{}_clusters.png".format(
            exp_name))
        self.plot_clustering(pred_clusters, filename)


class TwentyNewsGroups(DatasetBase):
    def __init__(self, max_df, min_df, doc_len_min, doc_len_max, classes):
        dataset = self._get_dataset(max_df, min_df, doc_len_min, doc_len_max, classes)
        self.X_tfidf = dataset['X_tfidf']
        self.X_count = dataset['X_count']
        self.X_bin = dataset['X_bin']
        self.y = dataset['y']
        self.target_names = dataset['target_names']
        self.terms = dataset['terms']

    def _get_dataset(self, max_df, min_df, doc_len_min, doc_len_max, classes):
        categories = ['alt.atheism', 'talk.religion.misc',
                      'comp.graphics', 'sci.space']
        categories = [categories[idx] for idx in classes]
        dataset = fetch_20newsgroups(subset='all', categories=categories,
                                     remove=('headers', 'footers', 'quotes'))

        # https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
        stemmer = PorterStemmer()
        analyzer = TfidfVectorizer().build_analyzer()
        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc)
                    if w not in ENGLISH_STOP_WORDS)

        # filter terms
        tvec = TfidfVectorizer(max_df=max_df, min_df=min_df, analyzer=stemmed_words)
        X_tfidf = tvec.fit_transform(dataset.data).toarray()
        terms = tvec.get_feature_names()
        logger.debug("terms=%s", terms)

        X_count = CountVectorizer(vocabulary=terms, analyzer=stemmed_words).fit_transform(
            dataset.data).toarray()
        X_bin = np.array(X_count > 0, int)
        assert X_tfidf.shape == X_count.shape == X_bin.shape
        doc_lengths = X_bin.sum(axis=1)
        logger.info("doc_lengths=%s", describe(doc_lengths))
        keep = (doc_lengths >= doc_len_min) & (doc_lengths <= doc_len_max)
        feat_lengths = X_bin[keep].sum(axis=0)
        logger.info("feat_lengths=%s", describe(feat_lengths))
        record = {
            'X_bin': X_bin[keep],
            'X_count': X_count[keep],
            'X_tfidf': X_tfidf[keep],
            'y': dataset.target[keep],
            'target_names': dataset.target_names,
            'terms': terms
        }
        logger.info("X=%s", record['X_bin'].shape)
        logger.info("y=%s", np.bincount(record['y']))
        return record

    def _get_words(self, pred_clusters):
        cluster_to_count = Counter(pred_clusters)
        logger.info("cluster_to_count=%s", cluster_to_count)
        for cluster_id, count in cluster_to_count.most_common():
            idxs, = np.where(pred_clusters == cluster_id)
            xis = self.X_tfidf[idxs]
            fdxs = np.argsort(xis.mean(axis=0))[-20:]
            top_terms = [self.terms[fdx] for fdx in fdxs]
            yield {
                'top_terms': top_terms,
                'cluster_size': count
            }

    def write_topics(self, pred_clusters, filename):
        with open(filename, 'w') as fwrite:
            for record in self._get_words(pred_clusters):
                line = ','.join([str(record['cluster_size'])] + record['top_terms'])
                fwrite.write(line + '\n')

    def evaluate_clusters(self, pred_clusters, exp_name, title):
        filename = os.path.expanduser("~/plot/20ng_{}_clusters.png".format(
            exp_name))
        self.plot_clustering(pred_clusters, filename, title)
        filename = os.path.expanduser("~/plot/20ng_{}_topics.csv".format(
            exp_name))
        self.write_topics(pred_clusters, filename)
