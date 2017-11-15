# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import scipy.sparse as sps
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import mmwrite

def check_matrix(X, format='csc', dtype=np.float32):
	if format == 'csc' and not isinstance(X, sps.csc_matrix):
		return X.tocsc().astype(dtype)
	elif format == 'csr' and not isinstance(X, sps.csr_matrix):
		return X.tocsr().astype(dtype)
	elif format == 'coo' and not isinstance(X, sps.coo_matrix):
		return X.tocoo().astype(dtype)
	elif format == 'dok' and not isinstance(X, sps.dok_matrix):
		return X.todok().astype(dtype)
	elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
		return X.tobsr().astype(dtype)
	elif format == 'dia' and not isinstance(X, sps.dia_matrix):
		return X.todia().astype(dtype)
	elif format == 'lil' and not isinstance(X, sps.lil_matrix):
		return X.tolil().astype(dtype)
	else:
		return X.astype(dtype)

class ISimilarity(object):
	"""Abstract interface for the similarity metrics"""

	def __init__(self, shrinkage=25):
		self.shrinkage = shrinkage

	def compute(self, X):
		pass


class Cosine(ISimilarity):

	def compute(self, X, Y, topn=50):
		

		X = check_matrix(X, 'csr', dtype=np.float32)
		Y = check_matrix(Y, 'csr', dtype=np.float32)

		Xsq = X.copy()
		norm = np.sqrt(Xsq.sum(axis=1))
		norm = np.asarray(norm).ravel()
		norm += 1e-6
		col_nnz = np.diff(X.indptr)
		X.data /= np.repeat(norm, col_nnz)

		Ysq = Y.copy()
		norm = np.sqrt(Ysq.sum(axis=1))
		norm = np.asarray(norm).ravel()
		norm += 1e-6
		col_nnz = np.diff(Y.indptr)
		Y.data /= np.repeat(norm, col_nnz)

		XT = X.T
		
		sim = sps.lil_matrix((Y.shape[0], X.shape[0]))


		#dot product on chunks and take topn
		chunk_size = 2000
		rest = Y.shape[0] % chunk_size
		chunk_num = Y.shape[0] // chunk_size

		for i in range(1,chunk_num):

			print("Started: ", i)

			temp = Y[(i-1)*chunk_size:i*chunk_size, :].dot(XT).toarray()

			
			idx_sorted = np.argsort(temp, axis=1)
			not_top_k = idx_sorted[:, :-topn]
			temp[np.arange(temp.shape[0])[:, None], not_top_k] = 0.0
			temp = sps.lil_matrix(temp)
			sim[(i-1)*chunk_size:i*chunk_size, :] = temp

			print("Finished for " + str(i))
		
		temp = Y[-rest:, :].dot(XT).toarray()
		idx_sorted = np.argsort(temp, axis=1)
		not_top_k = idx_sorted[:, :-topn]
		temp[np.arange(temp.shape[0])[:, None], not_top_k] = 0.0

		temp = sps.lil_matrix(temp)
		sim[-rest:, :] = temp
		

		sim = sim.tocsr()


		return sim

		


