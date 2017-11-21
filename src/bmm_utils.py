import numpy as np
from skbayes.mixture_models import VBBMM
from scipy.misc import logsumexp

def initialize(num_components=3):
	"""
	Initializes a bernoulli mixture model.
	"""

	model = VBBMM(n_components=num_components)

	return model

def train(model, 
		samples):
	"""
	Trains a bernoulli mixture model.
	"""

	return model.fit(samples)

def evaluate(model, 
			samples):
	"""
	Returns the log-likelihood assigned by a bernoulli mixture model.
	"""

	weights = model.weights_ # (n_components,)
	means = model.means_ # (n_features, n_components)

	log_weights = np.log(weights)
	log_prob_components = np.dot(samples, np.log(means)) + np.dot((1.-samples), np.log(1.-means))
	log_prob = logsumexp(log_weights + log_prob_components, axis=-1)

	return log_prob

def sample(model, 
			num_samples):
	"""
	Samples from a bernoulli mixture model.
	"""

	weights = model.weights_ # (n_components,)
	means = model.means_ # (n_features, n_components)

	samples = []

	for sample_idx in range(num_samples):
		component_idx = np.random.choice(len(weights), p=weights)
		sample = np.random.binomial(1, means[:, component_idx])
		samples.append(sample)

	samples = np.vstack(samples)

	return samples