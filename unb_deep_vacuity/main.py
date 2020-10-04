from processing_helper import text_to_tfidf_vectors
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

DATA_FILENAME = "dic_raw_0_0"

vectors = text_to_tfidf_vectors(DATA_FILENAME)
print(vectors.shape)
print(type(vectors))


path_to_file = "/home/dev/algoritmos/UnB/Deep Vacurity/unb_deep_vacuity/unb_deep_vacuity/resources/data/"
save_sparse_csr(path_to_file + "tfidf_vectors_0_0", vectors)
