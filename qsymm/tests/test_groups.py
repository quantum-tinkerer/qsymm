import pytest
import sympy
import numpy as np
import tinyarray as ta

from ..groups import PointGroupElement, Model, time_reversal, chiral, rotation


@pytest.mark.parametrize(
    "U,U_expected",
    [
        (1, np.eye(1)),
        ("1", np.eye(1)),
        ("kron(sigma_0, sigma_0)", np.eye(4)),
        (ta.array([[0, 1], [1, 0]]), 1 - np.eye(2))
    ]
)
def test_U_normalization(U, U_expected):
    R = np.eye(2)

    element = PointGroupElement(R, conjugate=False, antisymmetry=False, U=U)
    np.testing.assert_equal(element.R, R)
    np.testing.assert_equal(element.U, U_expected)