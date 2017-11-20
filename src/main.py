"""
Reference implementation for the paper.

Boosted Generative Models
Aditya Grover, Stefano Ermon
AAAI 2018

"""

import argparse
import numpy as np
from utils import *
from methods import *


def parse_args():

	parser = argparse.ArgumentParser(description='Boosted generative models.')
	
	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--datadir', nargs='?', default='data/',
						help='Input dataset directory')
	parser.add_argument('--dataset', nargs='?', default='nltcs',
						help='Dataset name.')
	parser.add_argument('--resultdir', nargs='?', default='results/',
						help='Directory for saving results.')

	parser.add_argument('--restore', default = False, action='store_true')

	# additive boosting options
	parser.add_argument('--run-add', default = False, action='store_true')
	parser.add_argument('--add-alpha', nargs='*', type=float, default=[1., 0.5, 0.25],
						help='')

	# generative boosting options
	parser.add_argument('--run-genbgm', default = False, action='store_true')
	parser.add_argument('--genbgm-alpha', nargs='*', type=float, default=[0.33, 0.33, 0.33],
						help='')
	parser.add_argument('--genbgm-beta', nargs='*', type=float, default=[0.125, 0.0625, 0.03125],
						help='')

	# discriminative boosting options
	parser.add_argument('--run-discbgm', default = False, action='store_true')	
	parser.add_argument('--discbgm-alpha', nargs='*', type=float, default=[1., 1.],
						help='')
	parser.add_argument('--discbgm-epochs', default=100, type=int,
					  help='number of epochs of training for every discriminator')
	parser.add_argument('--discbgm-burn-in', default=10000, type=int,
					  help='')

	# generative classification
	parser.add_argument('--run-classifier', default=False, action='store_true')
	
	return parser.parse_args()



def main():

	args = parse_args()
	args_dict = vars(args)
	globals().update(args_dict)

	np.random.seed(seed)
	np.set_printoptions(threshold=np.inf)

	train, test = load_data(dataset, path=datadir, splits=['train', 'test'])

	base_ll, baseline_model = eval_base(train, test, run_classifier)
	
	if run_add:
		add_ll = eval_add(train, test, add_alpha, run_classifier, baseline_model)
	
	if run_discbgm:
		discbgm_ll= eval_discbgm(train, test, discbgm_alpha, run_classifier, resultdir, baseline_model, seed, discbgm_epochs)

	if run_genbgm:
		genbgm_ll = eval_genbgm(train, test, genbgm_alpha, genbgm_beta, run_classifier, baseline_model)

if __name__ == '__main__':
	main()


