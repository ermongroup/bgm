"""
Reference implementation for the paper.

Boosted Generative Models
Aditya Grover, Stefano Ermon
AAAI Conference on Artificial Intelligence, 2018
"""

import argparse
import numpy as np
from utils import *
from methods import *


def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Boosted generative models.')
	
	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--datadir', nargs='?', default='data/',
						help='Input dataset directory')
	parser.add_argument('--dataset', nargs='?', default='nltcs',
						help='Dataset filename (in csv format)')
	parser.add_argument('--resultdir', nargs='?', default='results/',
						help='Directory for saving results')

	# additive boosting options
	parser.add_argument('--run-addbgm', default=False, action='store_true',
						help='Runs additive boosting if True')
	parser.add_argument('--addbgm-alpha', nargs='*', type=float, default=[0.5, 0.25],
						help='Model weights for additive boosting. \
						Length of list denotes number of boosting rounds')

	# generative boosting options
	parser.add_argument('--run-genbgm', default=False, action='store_true',
						help='Runs multiplicative generative boosting if True')
	parser.add_argument('--genbgm-alpha', nargs='*', type=float, default=[0.5, 0.5],
						help='Model weights for multiplicative generative boosting. \
						Length of list denotes number of boosting rounds')
	parser.add_argument('--genbgm-beta', nargs='*', type=float, default=[0.25, 0.125],
						help='Reweighting exponents for multiplicative generative boosting')

	# discriminative boosting options
	parser.add_argument('--run-discbgm', default=False, action='store_true',
						help='Runs multiplicative discriminative boosting if True')	
	parser.add_argument('--discbgm-alpha', nargs='*', type=float, default=[1., 1.],
						help='Model weights for multiplicative discriminative boosting. \
						Length of list denotes number of boosting rounds')
	parser.add_argument('--discbgm-epochs', default=100, type=int,
					    help='Number of epochs of training for every discriminator')
	parser.add_argument('--discbgm-burn-in', default=10000, type=int,
					    help='Burn in period for Markov chain sampling for multiplicative discriminative boosting')

	# generative classification
	parser.add_argument('--run-classifier', default=False, action='store_true',
						help='Performs generative classification if True')
	
	return parser.parse_args()


def main():

	args = parse_args()
	args_dict = vars(args)
	globals().update(args_dict)

	np.random.seed(seed)
	np.set_printoptions(threshold=np.inf)

	train, test = load_data(dataset, path=datadir, splits=['train', 'test'])

	base_ll, baseline_model = eval_base(train, test, run_classifier)
	
	if run_addbgm:
		addbgm_ll = eval_addbgm(train, test, addbgm_alpha, run_classifier, baseline_model)
	
	if run_discbgm:
		discbgm_ll= eval_discbgm(train, test, discbgm_alpha, run_classifier, resultdir, baseline_model, seed, discbgm_epochs)

	if run_genbgm:
		genbgm_ll = eval_genbgm(train, test, genbgm_alpha, genbgm_beta, run_classifier, baseline_model)

	return

if __name__ == '__main__':
	main()


