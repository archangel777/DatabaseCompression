from __future__ import division, print_function
import itertools

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy import linalg
from scipy.stats import multivariate_normal, normaltest
from sklearn import mixture
from sklearn.datasets import load_iris

import copy
import time

import findspark
findspark.init('C:\opt\spark\spark-2.2.1-bin-hadoop2.7')

from pyspark import SparkConf, SparkContext
conf = (SparkConf()
        .setMaster("local[*]")
        .setAppName("Teste")
        .set("spark.executor.memory", "2g"))
sc = SparkContext(conf = conf)

def silence_logger(sc):
    print('Silencing loggers...')
    sc.setLogLevel("FATAL")
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getRootLogger().setLevel( logger.Level.FATAL )
    logger.LogManager.getLogger("org").setLevel( logger.Level.FATAL )  
    logger.LogManager.getLogger("akka").setLevel( logger.Level.FATAL )

class StopWatch:
    def __init__(self):
        self.time = 0
        self.start_time = 0

    def reset(self):
        self.time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.time += time.time() - self.start_time


class MyModel:
    def __init__(self, n_gauss=5, err=0.01, data_size=64):
        self.err = err
        self.n_gauss = n_gauss
        self.data = None
        self.data_size = data_size
        # self.watches = {}

    def load_data(self, dataset='iris'):
        if dataset == 'iris':
            self.data = load_iris(return_X_y=True)[0]
        elif dataset == 'iris_complete':
            self.data = load_iris(return_X_y=True)
            aux = np.array([[i] for i in self.data[1]])
            self.data = np.append(self.data[0], aux, axis=1)
        elif dataset == 'tpc-h-5k':
            self.data = np.delete(np.genfromtxt('sampled_items.csv', delimiter='|'), 2, 1)
        elif dataset == 'tpc-h-full':
            self.data = np.delete(np.genfromtxt('sampled_items_full.csv', delimiter='|'), 2, 1)

        if self.data is not None:
            self.n = self.data.shape[0]
            self.m = self.data.shape[1]

    def test_normality(self, column):
        z, pval = normaltest(self.data[:, column])
        print(z, pval)
        if (pval < 0.055):
            print("Not normal distribution")
        else:
            print("Normaaaaal!")

    def train(self, show=False):
        if self.data is None:
            print('No data!')
            return
        self.gmm = mixture.GaussianMixture(n_components=self.n_gauss, max_iter=500).fit(self.data)
        self.mu = self.gmm.means_
        self.sigma = self.gmm.covariances_
        self.pi = self.gmm.weights_
        self.load_sigma_dots()
        self.anti_j = [np.arange(self.m) != j for j in range(self.m)]
        if show:
            print(self.gmm)

    def get_err_vec(self, data, completeRangeJ):
        return np.abs((completeRangeJ - data) / data)

    def find_min(self, data, completeRangeJ):
        errs = self.get_err_vec(data, completeRangeJ)
        min_err = min(errs)
        try:
            return min_err, np.where(errs == min_err)[0][0]
        except:
            return np.inf, -1

    def load_sigma_dots(self):
        sigma_inverses = np.array([
            [linalg.pinv(np.delete(np.delete(self.sigma[k], j, 0), j, 1)) for j in range(self.m)]
            for k in range(self.n_gauss)
        ])
        self.sigma_dots = np.array([
            [np.dot(np.delete(self.sigma[k, j], j), sigma_inverses[k, j]) for j in range(self.m)]
            for k in range(self.n_gauss)
        ])

    def reset_values(self):
        self.mvns = {}

        self.compact = np.zeros((self.n, self.m))
        self.totalErrs = np.zeros((self.n, self.m))

        self.t = np.zeros((self.n, self.n_gauss))
        self.mub = np.zeros((self.n, self.m, self.n_gauss))
        self.xb = np.zeros((self.n, self.m))

        self.minDifs = np.zeros((self.n, self.m))
        self.minDifIdxs = np.array([[-1 for j in range(self.m)] for i in range(self.n)])

        self.totalRems = 0

    def get_mub_helper(self):
        return np.array([
            [self.mu[k, j] - \
             np.dot(
                 self.sigma_dots[k, j], self.mu[k, self.anti_j[j]]
             )
             for j in range(self.m)]
            for k in range(self.n_gauss)])

    def get_mub(self, data):
        return np.array([
            [self.mub_helper[k, j] + \
             np.dot(
                 self.sigma_dots[k, j], data[self.anti_j[j]]
             )
             for j in range(self.m)]
            for k in range(self.n_gauss)])

    def get_t(self, X, tup_rangeJ):
        pdfs = [self.mvns[tup_rangeJ][k].pdf(X[~np.isnan(X)]) for k in range(self.n_gauss)]

        den = sum([self.pi[pos] * pdfs[pos] for pos in range(self.n_gauss)])
        return [self.pi[k] * pdfs[k] / den for k in range(self.n_gauss)]

    def predict_all(self, X, mub):
        rangeJ = np.argwhere(~np.isnan(X)).flatten()
        tup_rangeJ = tuple(rangeJ)

        # ---------------------------------------------------------
        if tup_rangeJ not in self.mvns:
            r_mu = [self.mu[k, rangeJ] for k in range(self.n_gauss)]
            r_sigma = [np.take(self.sigma[k, rangeJ], rangeJ, axis=1) for k in range(self.n_gauss)]

            self.mvns[tup_rangeJ] = [multivariate_normal(mean, cov) for mean, cov in zip(r_mu, r_sigma)]
        # ---------------------------------------------------------

        t = self.get_t(X, tup_rangeJ)

        return [sum([t[k] * mub[k, j] for k in range(self.n_gauss)]) for j in range(self.m)]

    def try_to_remove(self, X, data, mub):
        if all(np.isnan(X)): return X

        predictions = self.predict_all(X, mub)

        errs = self.get_err_vec(X, predictions)

        for j in np.argwhere(np.isnan(X)).flatten():
            predictions[j] = np.inf

        minErr, minErrIndex = self.find_min(data, predictions)
        # print minDif, minDifIndex
        if minErr > self.err or any(errs[np.argwhere(np.isnan(X)).flatten()] > self.err):
            return X

        X[minErrIndex] = np.nan

        return self.try_to_remove(X, data, mub)

    def assistParRun(self, data_tup):
        i, data = data_tup

        mub = self.get_mub(data)

        X = np.copy(data)
        return self.try_to_remove(X, data, mub)

    def parRun(self, prefix=''):
        self.reset_values()
        self.mub_helper = self.get_mub_helper()

        # self.watches['total_watch'].start()
        silence_logger(sc)
        self.totalRems = sc \
                .parallelize(list(enumerate(self.data))) \
                .map(lambda d: self.assistParRun(d)) \
                .map(lambda d: np.count_nonzero(np.isnan(d))) \
                .reduce(lambda x1, x2: x1 + x2)

        # v = Parallel(n_jobs=6)(delayed(assistParRun)(self, idx, self.data[idx], prefix) for idx in range(self.n))

        original_space = self.m * self.n * self.data_size
        data_space = (self.m * self.n - self.totalRems) * self.data_size + self.totalRems
        final_space = data_space + (self.m * self.n_gauss * (self.m + 1) + self.n_gauss) * self.data_size

        self.compression = (1 - data_space / original_space) * 100
        self.raw_compression = (1 - final_space / original_space) * 100

    def print_statistics(self):
        print('\n----------------------------------------------')
        print('\nTotal rems: {}'.format(self.totalRems))
        print('\nRem percentage: {:.2f}%'.format(self.totalRems / (self.n * self.m) * 100))
        print('\n----------------------------------------------')

    def print_times(self):
        print('\n----------------------------------------------')
        # print 'Total time: {:.2f}s'.format(self.watches['total_watch'].time)
        # for watch in [w for w in self.watches if w != 'total_watch']:
            # print '{} time: {:.2f}s'.format(watch, self.watches[watch].time)
            # print '{} time percentage: {:.2f}%'.format(watch, self.watches[watch].time / self.watches[
                # 'total_watch'].time * 100)
        print('----------------------------------------------')

    def print_final_space(self):
        print('\n----------------------------------------------')
        print('Compression: {:.2f}%'.format(self.compression))
        print('Compression without the matrices: {:.2f}%'.format(self.raw_compression))
        print('\n----------------------------------------------')
