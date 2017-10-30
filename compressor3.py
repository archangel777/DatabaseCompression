import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal
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

def train(data_dict, n_gauss, err=0.01, show=False):
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
	data_dict["err"] = err
	data_dict["mvns"] = {}

def get_mub_helper(data_dict):
	# data_dict["mu"] - 
	return np.array([
		[data_dict["mu"][k, j] - \
		np.dot(
			data_dict["sigma_dots"][k, j], data_dict["mu"][k, data_dict["anti_j"][j]]
		)
		for j in range(data_dict["m"])]
		for k in range(data_dict["n_gauss"])])

def get_mub(data_dict, i):
	return np.array([
		[data_dict["mub_helper"][k, j] + \
	 	np.dot(
		 	data_dict["sigma_dots"][k, j], data_dict["data"][i][data_dict["anti_j"][j]]
	 	)
		for j in range(data_dict["m"])]
		for k in range(data_dict["n_gauss"])])

def get_t(data_dict, i, rangeJ, tup_rangeJ):
	pdfs = [data_dict["mvns"][tup_rangeJ][k].pdf(data_dict["data"][i][rangeJ]) for k in range(data_dict["n_gauss"])]

	den = sum([data_dict["pi"][pos] * pdfs[pos] for pos in range(data_dict["n_gauss"])])
	return [data_dict["pi"][k] * pdfs[k] / den for k in range(data_dict["n_gauss"])]

def get_err_vec(data, completeRangeJ):
	return np.abs((completeRangeJ - data) / data)

def find_min(data, completeRangeJ):
	errs = get_err_vec(data, completeRangeJ)
	min_err = min(errs)
	try:
		return min_err, np.where(errs == min_err)[0][0]
	except:
		return np.inf, -1

# @numba.jit(int32(int32, int32))
# @numba.jit(parallel=True)
def parallel_run(i, data_dict, prefix):
	if i % (data_dict["n"] // 100) == 0 or i == data_dict["n"] - 1:
		print('{} {} / {}'.format(prefix, i + 1, data_dict["n"]),end='\r')

	nRem = 0
	mub = get_mub(data_dict, i)
	totalRems = 0
	minDifs = np.zeros(data_dict["m"])
	minDifIdxs = -np.ones(data_dict["m"])

	while (True):
		rangeJ = np.array([j for j in range(data_dict["m"]) if j not in minDifIdxs])
		tup_rangeJ = tuple(rangeJ)

		# ---------------------------------------------------------
		if tup_rangeJ not in data_dict["mvns"]:
			r_mu = [data_dict["mu"][k, rangeJ] for k in range(data_dict["n_gauss"])]
			r_sigma = [np.take(data_dict["sigma"][k, rangeJ], rangeJ, axis=1) for k in range(data_dict["n_gauss"])]

			data_dict["mvns"][tup_rangeJ] = [multivariate_normal(mean, cov) for mean, cov in zip(r_mu, r_sigma)]
		# ---------------------------------------------------------

		# ---------------------------------------------------------
		# Second bottleneck
		# self.watches['pred_watch'].start()
		t = get_t(data_dict, i, rangeJ, tup_rangeJ)
		# self.watches['pred_watch'].stop()
		# ---------------------------------------------------------
		xb = [sum([t[k] * mub[k, j] for k in range(data_dict["n_gauss"])]) for j in range(data_dict["m"])]
		# try:
		#	 self.xb[i] = copy.copy(xb2)
		# except Exception:
		#	 print("Error")

		# self.xb[i] = [sum([t[k]*mub[k, j] for k in range(self.n_gauss)]) for j in range(self.m)]

		erros = get_err_vec(data_dict["data"][i], xb)

		for j in range(data_dict["m"]):
			if j not in rangeJ:
				xb[j] = np.inf

		minDif, minDifIndex = find_min(data_dict["data"][i], xb)
		# print(type(minDif), type(minDifIndex))

		if minDif > data_dict["err"]:
			break

		isErr = False
		for j2 in minDifIdxs[minDifIdxs != -1]:
			if erros[j2] > data_dict["err"]:
				isErr = True

		if nRem + 1 >= data_dict["m"] or isErr:
			break

		minDifs[nRem] = minDif
		# self.minDifIdxs[i, nRem] = minDifIndex

		nRem += 1

		totalRems += 1
	return minDifIdxs, minDifs, totalRems

# @numba.jit(parallel=True)
def run(data_dict,n_jobs=6):
	data_dict["mub_helper"] = get_mub_helper(data_dict)
	prefix = ""
	for i in numba.prange(n_jobs):
		for idx in range(data_dict["n"]):
			parallel_run(int(data_dict["n"]/n_jobs * i + idx),data_dict,prefix)


data_dict = load_data("tpc-h-5k")
train(data_dict, 10)
run(data_dict)

