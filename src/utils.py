import os
import numpy as np
import csv
try:
	from time import perf_counter
except:
	from time import time
	perf_counter = time
import bmm_utils as model_utils
from scipy.misc import logsumexp
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def modify_ext(path='./data/'):
	for file in os.listdir(path):
		print(file)
		if '.ts.' in file:
			new_file = file.replace('.ts.', '.train.')
			print(new_file)
			os.rename(os.path.join(path, file), os.path.join(path, new_file))

# Utils for loading data

def csv_2_numpy(filename, 
				path, 
				sep=',', 
				type='int32'):
	"""
	Utility to read a dataset in csv format into a numpy array
	"""

	file_path = os.path.join(path, filename)
	reader = csv.reader(open(file_path, "r"), delimiter=sep)
	x = list(reader)
	array = np.array(x).astype(type)
	return array

def load_data(dataset_name, 
			   path,
			   sep=',',
			   type='int32',
			   suffix='data',
			   splits=['train', 'valid', 'test'],
			   verbose=True):
	"""
	Loading splits by suffix from csv files
	"""

	csv_files = ['{0}.{1}.{2}'.format(dataset_name, ext, suffix) for ext in splits]

	load_start_t = perf_counter()
	dataset_splits = [csv_2_numpy(file, path, sep, type) for file in csv_files]
	load_end_t = perf_counter()

	if verbose:
		print('Dataset splits for {0} loaded in {1:.2f} secs'.format(dataset_name,
																 load_end_t - load_start_t))
		for data, split in zip(dataset_splits, splits):
			print('\t{0}:\t{1}'.format(split, data.shape))

	return dataset_splits

# Utils for boosting
def get_add_reweighted_training_set(ensemble, 
									alphas, 
									train_samples):

	log_likelihood = get_add_boosted_log_likelihood(ensemble, alphas, train_samples)
	log_uweights = -1*log_likelihood
	weights = np.exp(log_uweights-logsumexp(log_uweights))
	num_examples = train_samples.shape[0]
	reweighted_idx = np.random.choice(num_examples, num_examples, p=weights, replace=True)
	reweighted_samples = train_samples[reweighted_idx]
	
	return reweighted_samples


def get_add_boosted_log_likelihood(ensemble, 
									alphas, 
									samples):

	num_examples = samples.shape[0]
	log_likelihood = np.zeros((num_examples))
	for model_idx, model in enumerate(ensemble):
		if model_idx == 0:
			log_likelihood = model_utils.evaluate(model, samples)
		else:
			additive_log_models = np.array([np.log(1-alphas[model_idx])+log_likelihood, np.log(alphas[model_idx])+model_utils.evaluate(model, samples)])
			log_likelihood = logsumexp(additive_log_models, axis=0)
	return log_likelihood

def get_multiply_boosted_unnormalized_log_likelihood(ensemble, 
													alphas, 
													samples, 
													genbgm=True):

	if genbgm:
		num_examples = samples.shape[0]
		ull = np.zeros((num_examples)) # unnormalized log-likelihood
		for model_idx, model in enumerate(ensemble):
			model_contrib = alphas[model_idx] * model_utils.evaluate(model, samples)
			ull += model_contrib
	else:
		ull = alphas[0] * model_utils.evaluate(ensemble[0], samples)
		for model_idx in range(1, len(ensemble)):
			ull += alphas[model_idx] * ensemble[model_idx].predict_log_odds(samples)

	return ull


def get_log_partition_estimate(ensemble, 
								alphas, 
								proposal, 
								genbgm=True, 
								num_samples=1000000):

	import bmm_utils
	proposal_samples = bmm_utils.sample(proposal, num_samples)
	ull = get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, proposal_samples, genbgm)
	log_importance_weights = bmm_utils.evaluate(proposal, proposal_samples)
	logZ = logsumexp(ull-log_importance_weights) - np.log(num_samples)

	return logZ

def get_genbgm_reweighted_training_set(ensemble, 
										alphas, 
										train_samples, 
										betas):

	num_examples = train_samples.shape[0]
	ull = get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, train_samples, True)
	neg_ull = -1 * ull
	weight_ull = neg_ull*betas[len(ensemble)-1]
	weights = np.exp(weight_ull - logsumexp(weight_ull))
	reweighted_idx = np.random.choice(num_examples, num_examples, p=weights, replace=True)
	reweighted_samples = train_samples[reweighted_idx]

	return reweighted_samples

def get_discbgm_reweighted_training_set(ensemble, 
										alphas, 
										train_samples):

	num_examples = train_samples.shape[0]
	ull = get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, train_samples, False)
	neg_ull = -1 * ull
	weight_ull = neg_ull
	weights = np.exp(weight_ull - logsumexp(weight_ull))
	reweighted_idx = np.random.choice(num_examples, num_examples, p=weights, replace=True)
	reweighted_samples = train_samples[reweighted_idx]

	return reweighted_samples

def get_next_state(current_states):

	proposal_states = current_states.copy()
	sample_idx = np.random.randint(current_states.shape[1], size=(proposal_states.shape[0],))
	proposal_states[np.arange(proposal_states.shape[0]), sample_idx] = 1-current_states[np.arange(proposal_states.shape[0]), sample_idx]

	return proposal_states

def get_model_samples(ensemble, 
					 alphas, 
					 init_states):

	# MCMC sampling for discrete state spaces

	burn_in = 10000
	current_states = init_states
	current_log_densities = get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, current_states, genbgm=False)

	display_step = burn_in/2

	ones_mask = np.ones((init_states.shape[0],))
	zeros_mask = np.zeros((init_states.shape[0],))

	for step in range(burn_in):
		if (step+1)%display_step==0:
			print('Step', step+1)
		proposal_states = get_next_state(current_states)
		proposal_log_densities = get_multiply_boosted_unnormalized_log_likelihood(ensemble, alphas, proposal_states, genbgm=False)
		diff_loss = -1 * (proposal_log_densities - current_log_densities)        

		next_state_mask = np.where(np.logical_or(diff_loss<0, np.random.random_sample(size=diff_loss.shape) 
			< np.exp(-1 * diff_loss)), ones_mask, zeros_mask)
		next_state_mask = np.expand_dims(next_state_mask, axis=1)
		if len(init_states.shape) == 3:
			next_state_mask = np.expand_dims(next_state_mask, axis=1)
		current_states = next_state_mask*proposal_states + (1-next_state_mask)*current_states

	return current_states


class DiscriminativeModel(object):

	def __init__(self, 
				sess, 
				seed, 
				disc_id, 
				n_input):

		tf.set_random_seed(seed)
		self.sess = sess

		self.n_input = n_input
		self.n_outputs = 1
		self.arch = [100, 100]

		self.disc_id = 'disc' + str(disc_id)
		self.lr=1e-4 if disc_id == 1 else 5e-5 
		self.optimizer = tf.train.AdamOptimizer

		self.x = tf.placeholder(tf.float32, [None, self.n_input], name='x') # create symbolic variables
		self.y = tf.placeholder(tf.float32, [None, self.n_outputs], name='y')

		self.dropout_rate = 0. 
		self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

		self.output_logits = self.run_network(self.x, reuse=None)
		self.loss = self.create_loss(self.output_logits, self.y)

		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss, 
			global_step=self.global_step)
		self.saver = tf.train.Saver(max_to_keep=None)
		self.init_op = tf.global_variables_initializer()
	   

	def run_network(self, 
					x, 
					is_training=False, 
					reuse=None):

		activations = x
		with tf.variable_scope(self.disc_id):
			for layer_idx, layer_dim in enumerate(self.arch):
				activations = tf.layers.dense(activations, layer_dim, 
					activation=tf.nn.relu, reuse=reuse, name='dense-'+str(layer_idx))
				activations = tf.layers.dropout(activations, training=is_training, name='dropout-'+str(layer_idx))

			output_logits = tf.layers.dense(activations, 1, 
					activation=None, reuse=reuse, name='dense-output')

		return output_logits

	def create_loss(self, 
					logits, 
					y):

		y = self.y
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
		self.ll = logits

		return loss

	def predict_log_odds(self, 
						X):

		log_odds = self.sess.run(self.ll, feed_dict={self.x: X})

		return np.squeeze(log_odds)

	def train(self, 
			trX, 
			trY, 
			valX=None, 
			valY=None, 
			num_epochs=100,
			savedir=None):

		sess = self.sess
		sess.run(self.init_op)
		if valX is None:
			valX = trX[int(0.8*trX.shape[0]):, :]
			trX = trX[:int(0.8*trX.shape[0]), :]
		if valY is None:
			valY = trY[int(0.8*trY.shape[0]):, :]
			trY = trY[:int(0.8*trY.shape[0]), :]
	   
		global_steps = []
		validation_losses = []
		save_paths = []
		num_epochs = 100
		for i in range(num_epochs):
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
				sess.run(self.train_op, feed_dict={self.x: trX[start:end], self.y: trY[start:end]})
			train_loss, gs = sess.run([self.loss, self.global_step], feed_dict={self.x: trX, self.y: trY, self.is_training: True})
			if savedir is not None:
				save_path=self.saver.save(sess, os.path.join(savedir,'model.ckpt'), global_step=gs)
				global_steps.append(gs)
				save_paths.append(save_path)
			validation_loss = sess.run(self.loss, feed_dict={self.x: valX, self.y: valY, self.is_training: False})
			if i%20 == 0:
				print('Epoch:', i, 'train loss:', train_loss, 'validation loss:', validation_loss)

			validation_losses.append(validation_loss)
		min_idx = validation_losses.index(min(validation_losses))
		print('Restoring ckpt with lowest validation error: ', save_paths[min_idx])
		self.saver.restore(sess, save_paths[min_idx])

		return

