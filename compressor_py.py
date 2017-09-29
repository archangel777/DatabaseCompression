
# coding: utf-8

# In[2]:

from __future__ import division
import itertools

import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from sklearn import mixture
from sklearn.datasets import load_iris
get_ipython().magic(u'matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
np.seterr(divide='ignore', invalid='ignore')


# In[91]:

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
        
    def load_data(self, dataset='iris'):
        if dataset == 'iris':
            self.data = load_iris(return_X_y=True)[0]
        elif dataset == 'iris_complete':
            self.data = load_iris(return_X_y=True)
            aux = np.array([[i] for i in self.data[1]])
            self.data = np.append(self.data[0], aux, axis=1)
        elif dataset == 'tpc-h':
            self.data = np.delete(np.genfromtxt('sampled_items.csv', delimiter='|'), 2, 1)
        
        if self.data is not None:
            self.n = self.data.shape[0]
            self.m = self.data.shape[1]
        
    def train(self, show=False):
        if self.data is None:
            print 'No data!'
            return
        self.gmm = mixture.GaussianMixture(n_components=self.n_gauss, max_iter=500).fit(self.data)
        self.mu = self.gmm.means_
        self.sigma = self.gmm.covariances_
        self.pi = self.gmm.weights_
        self.load_sigma_dots()
        self.anti_j = [np.arange(self.m) != j for j in range(self.m)]
        if show:
            print self.gmm
        
    def get_err_vec(self, data, completeRangeJ):
        return np.abs((completeRangeJ - data)/ data)

    def find_min(self, data, completeRangeJ):
        errs = self.get_err_vec(data, completeRangeJ)
        min_err = min(errs)
        try:
            return min_err, np.where(errs == min_err)[0][0]
        except:
            return np.inf, -1
        
    def load_sigma_dots(self):
        sigma_inverses = np.array(  [
                            [linalg.pinv(np.delete(np.delete(self.sigma[k], j, 0), j, 1)) for j in range(self.m)] 
                            for k in range(self.n_gauss)
                        ])
        self.sigma_dots = np.array(  [
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
        
    def get_mub(self, i):
        return np.array([
                    [self.mub_helper[k, j] + \
                        np.dot(
                            self.sigma_dots[k, j], self.data[i, self.anti_j[j]]
                        ) 
                    for j in range(self.m)]
                for k in range(self.n_gauss)])
    
    def get_t(self, i, rangeJ, tup_rangeJ):
        pdfs = [self.mvns[tup_rangeJ][k].pdf(self.data[i][rangeJ]) for k in range(self.n_gauss)]
        
        den = sum([self.pi[pos]*pdfs[pos] for pos in range(self.n_gauss)])
        return [self.pi[k]*pdfs[k] / den for k in range(self.n_gauss)]
    
    def run(self):
        self.reset_values()
        self.mub_helper = self.get_mub_helper()
        
        self.total_watch = StopWatch()
        self.pred_watch = StopWatch()
        self.aux_watch = StopWatch()

        self.total_watch.start()
        for i in range(self.n):
            if i%(self.n//100) == 0 or i == self.n-1:
                print '\r',
                print i,

            nRem = 0

            #--------------------------------------------------------- 
            # First bottleneck
            self.aux_watch.start()
            mub = self.get_mub(i)
            self.aux_watch.stop()
            #--------------------------------------------------------- 

            while(True):
                
                rangeJ = np.array([j for j in range(self.m) if j not in self.minDifIdxs[i]])
                tup_rangeJ = tuple(rangeJ)
                
                #---------------------------------------------------------
                if tup_rangeJ not in self.mvns:
                    r_mu = [self.mu[k, rangeJ] for k in range(self.n_gauss)]
                    r_sigma = [np.take(self.sigma[k, rangeJ], rangeJ, axis=1) for k in range(self.n_gauss)]

                    self.mvns[tup_rangeJ] = [multivariate_normal(mean, cov) for mean, cov in zip(r_mu, r_sigma)]
                #---------------------------------------------------------

                #--------------------------------------------------------- 
                # Second bottleneck
                self.pred_watch.start()
                t = self.get_t(i, rangeJ, tup_rangeJ)
                self.pred_watch.stop()
                #---------------------------------------------------------
                self.xb[i] = [sum([t[k]*mub[k, j] for k in range(self.n_gauss)]) for j in range(self.m)]
                
                erros = self.get_err_vec(self.data[i], self.xb[i])
            
                completeRangeJ = self.xb[i]
                for j in range(self.m):
                    if j not in rangeJ:
                        completeRangeJ[j] = np.inf
                
                minDif, minDifIndex = self.find_min(self.data[i], completeRangeJ)
                #print minDif, minDifIndex
                if minDif > self.err:
                    break
                    
                isErr = False
                for j2 in [j for j in self.minDifIdxs[i] if j != -1]:
                    if erros[j2] > self.err:
                        isErr = True

                if nRem + 1 >= self.m or isErr:
                    break

                self.minDifs[i, nRem] = minDif
                self.minDifIdxs[i, nRem] = minDifIndex

                nRem += 1

                self.totalErrs[i] = self.minDifs[i]
                self.totalRems += 1

            self.compact[i] = self.minDifIdxs[i]

        self.total_watch.stop()
        
        original_space = self.m*self.n*self.data_size
        data_space = (self.m * self.n - self.totalRems) * self.data_size + self.totalRems
        final_space = data_space + (self.m * self.n_gauss * (self.m + 1) + self.n_gauss) * self.data_size
        
        self.relative_data_space = data_space/original_space * 100
        self.relative_final_space = final_space/original_space * 100
        
    def print_statistics(self):
        print '\n----------------------------------------------'
        print '\nTotal rems: {}'.format(self.totalRems)
        print 'Rem percentage: {:.2f}%'.format(self.totalRems/(self.n*self.m) * 100)
        print '\n----------------------------------------------'
    
    def print_times(self):
        print '\n----------------------------------------------'
        print 'Total time: {:.2f}s'.format(self.total_watch.time)
        print 'Prediction time: {:.2f}s'.format(self.pred_watch.time)
        print 'Prediction time percentage: {:.2f}%'.format(self.pred_watch.time/self.total_watch.time * 100)
        print 'Aux time: {:.2f}s'.format(self.aux_watch.time)
        print 'Aux time percentage: {:.2f}%'.format(self.aux_watch.time/self.total_watch.time * 100)
        print '----------------------------------------------'
        
    def print_final_space(self):
        print '\n----------------------------------------------'
        print 'Percentage relative to original: {:.2f}%'.format(self.relative_final_space)
        print 'Percentage without the matrices: {:.2f}%'.format(self.relative_data_space)
        print '\n----------------------------------------------'


# In[92]:

model = MyModel(n_gauss=30, err=0.01)
#model.load_data('iris_complete')
#model.load_data('iris')
model.load_data('tpc-h')

model.train()

model.run()

model.print_times()

#model.print_statistics()

#model.print_final_space()


# In[93]:

def plot_gauss(n, dataset, e):
    x = [i+1 for i in range(n)]
    y = []
    for i in x:
        model = MyModel(n_gauss=i, err=e)
        model.load_data(dataset)
        #model.load_data('iris')
        #model.load_data('tpc-h')

        model.train()
        model.run()
        y.append(model.relative_final_space)
    
    plt.plot(x, y)
    plt.show()

plot_gauss(20, 'tpc-h', 0.01)
        


# In[ ]:



