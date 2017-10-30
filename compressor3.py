import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# mpl.use('Agg')

from sklearn import mixture
from sklearn.datasets import load_iris
from joblib import Parallel, delayed

import numba

def load_data(dataset='iris'):
	data_dict = {}
	if dataset == 'iris':
		data_dict["data"] = load_iris(return_X_y=True)[0]
	elif dataset == 'iris_complete':
		data_dict["data"] = load_iris(return_X_y=True)
		aux = np.array([[i] for i in data_dict.data[1]])
		data_dict["data"] = np.append(data_dict.data[0], aux, axis=1)
	elif dataset == 'tpc-h-5k':
		data_dict["data"] = np.delete(np.genfromtxt('sampled_items.csv', delimiter='|'), 2, 1)
	elif dataset == 'tpc-h-full':
		data_dict["data"] = np.delete(np.genfromtxt('sampled_items_full.csv', delimiter='|'), 2, 1)

	if data_dict["data"] is not None:
		data_dict["n"],data_dict["m"] = data_dict["data"].shape

	return data_dict

def load_sigma_dots(data_dict,n_gauss):
	sigma_inverses = np.array([
		[np.linalg.pinv(np.delete(np.delete(data_dict["sigma"][k], j, 0), j, 1)) for j in range(data_dict["m"])]
		for k in range(n_gauss)
	])
	sigma_dots = np.array([
		[np.dot(np.delete(data_dict["sigma"][k, j], j), sigma_inverses[k, j]) for j in range(data_dict["m"])]
		for k in range(n_gauss)
	])

	return sigma_dots

def train(data_dict,n_gauss, show=False):
	if data_dict["data"] is None:
		print('No data!')
		return

	data_dict["gmm"] = mixture.GaussianMixture(n_components=n_gauss, max_iter=500).fit(data_dict["data"])

	data_dict["mu"] = data_dict["gmm"].means_
	data_dict["sigma"] = data_dict["gmm"].covariances_
	data_dict["pi"] = data_dict["gmm"].weights_
	data_dict["sigma_dots"] = load_sigma_dots(data_dict,n_gauss)
	data_dict["anti_j"] = [np.arange(data_dict["m"]) != j for j in range(data_dict["m"])]
	data_dict["n_gauss"] = n_gauss

def get_mub_helper(data_dict):
	# data_dict["mu"] - 
	return np.array([
		[data_dict["mu"][k, j] - \
		np.dot(
			data_dict["sigma_dots"][k, j], data_dict["mu"][k, data_dict["anti_j"][j]]
		)
		for j in range(data_dict["m"])]
		for k in range(data_dict["n_gauss"])])


# @numba.jit(int32(int32, int32))
@numba.jit(nopython=True, parallel=True)
def parallel_run(i, data_dict, prefix):
	if i % (data_dict["n"] // 100) == 0 or i == data_dict["n"] - 1:
		print('{} {} / {}'.format(prefix, i + 1, data_dict["n"]),end='\r')

	nRem = 0
	mub = get_mub(data_dict["data"])
	totalRems = 0
	minDifs = np.zeros(data_dict["m"])
	minDifIdxs = -np.ones(data_dict["m"])

	while (True):
		rangeJ = np.array([j for j in range(data_dict["m"]) if j not in minDifIdxs])
		tup_rangeJ = tuple(rangeJ)

@numba.jit(parallel=True)
def run(data_dict,n_jobs=6):
	data_dict["mub_helper"] = get_mub_helper(data_dict)
	prefix = ""
	for i in numba.prange(n_jobs):
		for idx in range(data_dict["n"])
			parallel_run(idx,data_dict,prefix)


data_dict = load_data()
train(data_dict,10)
run(data_dict)

