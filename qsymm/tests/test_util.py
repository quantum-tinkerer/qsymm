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
    S1 = PointGroupElement(sympy.eye(2), False, False,
                           np.eye(3))
    S2 = PointGroupElement(sympy.Matrix([[0, 1], [1, 0]]), True, False,
                           np.eye(3))
    S3 = PointGroupElement(np.eye(2), False, False,
                           1j * np.eye(3))
    C6s = PointGroupElement(sympy.ImmutableMatrix(
                                [[sympy.Rational(1, 2), sympy.sqrt(3)/2],
                                 [-sympy.sqrt(3)/2,       sympy.Rational(1, 2)]]
                                                 ))
    C6f = PointGroupElement(np.array(
                                    [[1/2, np.sqrt(3)/2],
                                     [-np.sqrt(3)/2, 1/2]]
                                                     ))

    assert S2**2 == S1
    assert not S1 == S2
    assert S1 == S3
    assert C6s == C6f
    # Mixing sympy with other types raises an error
    with raises(ValueError):
        S = C6s * C6f
