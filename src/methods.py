import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from bmm_utils import *
from utils import *
from scipy.misc import logsumexp

def predict(test, 
	ensemble, 
	alphas, 
	log_likelihood_evaluator, 
	genbgm=None):
	"""
	Generative classification for predicting one variable given the rest.
	"""

	print('Running generative classifier...')
	num_samples = test.shape[0]
	num_dim = test.shape[1]
	all_zeros = np.zeros(num_samples)
	all_ones = np.ones(num_samples)

	avg_xent = 0.
	avg_accuracy = 0.
	for dim_idx in range(num_dim):
		train_data0 = test.copy()
		train_data0[:, dim_idx] = all_zeros

		train_data1 = test.copy()
		train_data1[:, dim_idx] = all_ones

		if genbgm is None:
			unnormalized_log_p0 = log_likelihood_evaluator(ensemble, alphas, train_data0)
			unnormalized_log_p1 = log_likelihood_evaluator(ensemble, alphas, train_data1)
		else:
			unnormalized_log_p0 = log_likelihood_evaluator(ensemble, alphas, train_data0, genbgm=genbgm)
			unnormalized_log_p1 = log_likelihood_evaluator(ensemble, alphas, train_data1, genbgm=genbgm)

		log_p1 = unnormalized_log_p1 - logsumexp(np.array([unnormalized_log_p0, unnormalized_log_p1]), axis=0)
		log_p0 = np.log(1-np.exp(log_p1))

		log_p1 = log_p1.reshape([-1,1])
		log_p0 = log_p0.reshape([-1,1])

		predicted_log_prob = np.hstack((log_p0, log_p1))
		true_labels = test[np.arange(num_samples), [dim_idx]]
		xent = -1 * np.mean(true_labels*predicted_log_prob[:,1] + (1.-true_labels)*predicted_log_prob[:,0])
		avg_xent += xent

		accuracy = np.mean(np.equal(np.argmax(predicted_log_prob, axis=1), true_labels))
		avg_accuracy += accuracy
	avg_xent /= num_dim
	avg_accuracy /= num_dim
	print('Average xent loss: %.2f, accuracy: %.2f' % (avg_xent, avg_accuracy))

	return avg_accuracy



def eval_base(train, 
	test, 
	run_classifier=False):
	"""
	Trains and evaluates the base generative model.
	"""

	print('Running base generative model...')
	model = initialize()
	model.fit(train)
	ll_test = np.mean(get_add_boosted_log_likelihood([model], [1.], test))
	print('Log-likelihood for base generative model: %.2f' % ll_test)

	if run_classifier:
		pred_accuracy = predict(test, [model], [1.], get_add_boosted_log_likelihood)
	print()

	return ll_test, model

def eval_addbgm(train, 
	test, 
	alphas, 
	run_classifier, 
	baseline_model=None):
	"""
	Trains and evaluates the additive boosting model ensemble.
	"""

	print('Running additive boosting...')
	ensemble = []
	for t in range(len(alphas)):
		print('Round', t)
		if t == 0 and baseline_model is not None:
			model = baseline_model
		else:
			if t == 0:
				reweighted_samples = train
			else:
				reweighted_samples = get_add_reweighted_training_set(ensemble, alphas, train)
			model = initialize()
			model.fit(reweighted_samples)
		ensemble.append(model)
	ll_test = np.mean(get_add_boosted_log_likelihood(ensemble, alphas, test))
	print('Log-likelihood with additive boosting after %d rounds: %.2f' % (len(alphas), ll_test))

	if run_classifier:
		predict(test, ensemble, alphas, get_add_boosted_log_likelihood)
	print()

	return ll_test

def eval_discbgm(train,
	test,
	alphas, 
	run_classifier, 
	resultdir, 
	baseline_model=None,
	seed=0, 
	num_epochs=100):
	"""
	Trains and evaluates the multiplicative discriminative boosting model ensemble.
	"""

	print('Running discriminative multiplicative boosting...')
	tf.reset_default_graph()
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	if not os.path.exists(resultdir):
		os.makedirs(resultdir)

	print('Round 0')
	if baseline_model is not None:
		model = baseline_model
	else:
		model = initialize()
		model.fit(train)
	ensemble = [model]
	
	for t in range(1, len(alphas)):
		disc = DiscriminativeModel(sess, seed, t, train.shape[1])
		ensemble.append(disc)

	for t in range(1, len(alphas)):
		print('Round', t)
		if t==1:
			model_samples = sample(ensemble[0], train.shape[0])
		else:
			model_samples = get_model_samples(ensemble[:t], alphas, train)
		x = np.vstack((train, model_samples))
		y = np.vstack((np.ones((train.shape[0],1)), np.zeros((model_samples.shape[0],1))))
		perm = np.arange(x.shape[0])
		np.random.shuffle(perm)
		x = x[perm]
		y = y[perm]
		ensemble[t].train(x, y, num_epochs=num_epochs, savedir=resultdir)

	print('Estimating normalization constant for boosted ensemble...')
	proposal = ensemble[0]
	ull_test = np.mean(get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, test, False))
	logZ_estimate = get_log_partition_estimate(ensemble, alphas, proposal, False)
	ll_test = ull_test-logZ_estimate
	print('Log-likelihood with discriminative boosting after %d rounds: %.2f' % (len(alphas), ll_test))

	if run_classifier:
		predict(test, ensemble, alphas, get_multiply_boosted_unnormalized_log_likelihood, genbgm=False)
	print()

	return ll_test

def eval_genbgm(train, 
	test,
	alphas,
	betas,
	run_classifier, 
	baseline_model):
	"""
	Trains and evaluates the multiplicative generative boosting model ensemble.
	"""

	print('Running generative multiplicative boosting...')
	ensemble = []
	for t in range(len(alphas)):
		print('Round', t)
		if t == 0 and baseline_model is not None:
			model = baseline_model
		else:
			if t == 0:
				reweighted_samples = train
			else:
				reweighted_samples = get_genbgm_reweighted_training_set(ensemble, alphas, train, betas)
			model = initialize()
			model.fit(reweighted_samples)
		ensemble.append(model)

	print('Estimating normalization constant for boosted ensemble...')
	proposal = ensemble[0]
	ull_test = np.mean(get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, test, True))
	logZ_estimate = get_log_partition_estimate(ensemble, alphas, proposal, True)
	ll_test = ull_test-logZ_estimate
	print('Log-likelihood with generative boosting after %d rounds: %.2f' % (len(alphas), ll_test))
	
	if run_classifier:
		predict(test, ensemble, alphas, get_multiply_boosted_unnormalized_log_likelihood, genbgm=True)
	print()

	return ll_test