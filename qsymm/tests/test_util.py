import pytest
import warnings
import sympy
import itertools
import numpy as np
import scipy.linalg as la

from pytest import raises
from ..linalg import matrix_basis, sparse_basis
from ..groups import PointGroupElement


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
    
    
def test_spatial_types():
    S1 = PointGroupElement(sympy.eye(2), True, True,
                           np.random.rand(3, 3))
    S2 = PointGroupElement(sympy.Matrix([[0, 1], [1, 0]]), True, False,
                           np.random.rand(3, 3))
    S3 = PointGroupElement(np.eye(2), False, False,
                           np.random.rand(3, 3))
    S4 = PointGroupElement(np.array([[0, 1.2], [1.5, 0]]), True, False,
                           np.random.rand(3, 3))
    
    # Multiplying or comparing objects allowed if both or neither have sympy spatial parts
    S = S1 * S2
    assert not S1 == S2
    S = S3 * S4
    assert not S3 == S4
    # Mixing sympy with other types raises an error
    with raises(ValueError, message="Multiplying PointGroupElements only allowed "
                                    "if neither or both have sympy spatial parts R."):
        S = S1 * S3
    with raises(ValueError, message="Comparing PointGroupElements only allowed "
                                    "if neither or both have sympy spatial parts R."):
        S4 == S2
        