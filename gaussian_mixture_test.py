from sklearn.mixture import BayesianGaussianMixture
import numpy as np

def weighted_mean(values, weights):
	return sum(values*weights)/sum(weights)

def cut_matrix(matrix, indexes):
	return np.delete(np.delete(matrix, indexes, axis=1), indexes, axis=2)

def predict(model, X):
	predicted = np.copy(X)
	w = model.weights_
	means = model.means_
	cov = model.covariances_
	precisions = model.precisions_

	missing_indexes = np.argwhere(X != None)[0]
	for i in missing_indexes:
		values = [mean[i] for mean in means]
		predicted[i] = weighted_mean(values, w)
	return predicted


mix = BayesianGaussianMixture(2)

data = np.array([	[1, 2, 4],
					[2, 3, 5],
					[3, 3, 6],
					[4, 5, 6],
					[2, 5, 10]])

mix.fit(data)

print(predict(mix, np.array([3, 3, None])))