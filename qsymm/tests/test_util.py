import kwant
import pytest
import warnings
import sympy
import itertools
import numpy as np
import scipy.linalg as la

from ..linalg import matrix_basis, sparse_basis


def test_sparse_basis():
    reals = [True, False]
    for _, real in itertools.product(range(3), reals):
        dim = np.random.randint(5, 10)
        H = np.random.rand(dim, dim)
        U = la.expm(1j*(H+H.T.conj()))
        # Sparsify a full rank matrix
        sparse_U = sparse_basis(U, num_digits=4, reals=real)
        assert sparse_U.shape == U.shape
        # Make one row linearly dependent on the others
        U[-1, :] = sum([np.random.rand()*row for row in U[:-1, :]])
        with warnings.catch_warnings(record=True) as w:
            sparse_U = sparse_basis(U, num_digits=4, reals=real)
            assert sparse_U.shape[0] == dim-1
            assert len(w) == 1 # A warning should be raised
