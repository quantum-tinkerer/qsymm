import pytest
import sympy
import numpy as np
import tinyarray as ta

from ..groups import PointGroupElement, Model, time_reversal, chiral, rotation, generate_group, cubic
from ..linalg import allclose
from ..characters import SpaceGroupElement, LittleGroupElement, PointGroup, SpaceGroup, LittleGroup


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

def test_SGE_LGE():
    R1 = rotation(1/4, [0, 0, 1])
    R2 = rotation(1/2, [1, 0, 0])

    S1 = SpaceGroupElement(R1, t=[0, 0, 1/4], periods=np.eye(3))
    S2 = SpaceGroupElement(R2, t=[0, 0, 0], periods=np.eye(3))
    SG = generate_group([S1, S2])

    np.testing.assert_equal(len(SG), 8)

    np.testing.assert_equal(S1 * S1**(-1), S1.identity())
    assert allclose((S1 * S2).t, [0, 0, 1/4])
    assert allclose((S1**(-1)).t, [0, 0, -1/4])
    assert not S1 == S2

    L1 = LittleGroupElement(S1, k=[0, 0, 1/2], phase=1)
    L2 = LittleGroupElement(S2, k=[0, 0, 1/2], phase=1)

    assert allclose((L1 * L2).t, [0, 0, 1/4])
    assert allclose((L1 * L2).phase, 1)
    assert allclose((L1**4).phase, -1)
    assert allclose((L1 * (L1**(-1))).phase, 1)
    np.testing.assert_equal(L1 * L1**(-1), L1.identity())
    assert not L1 == L2

    assert type(S1) == SpaceGroupElement

def test_pointgroup():
    pg = PointGroup(cubic(tr=False, ph=False, generators=True, double_group=False))
    irrep_dims = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    assert allclose(pg.character_table[:, 0], irrep_dims)
    irreps = pg.irreps()
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality() for i in irreps], np.ones((10,)))

    pg = PointGroup(cubic(tr=False, ph=False, generators=True, double_group=True))
    irrep_dims = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    assert allclose(pg.character_table[:, 0], irrep_dims)
    irreps = pg.irreps()
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality() for i in irreps], [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1])

    # Test phase fixing
    g = cubic(tr=False, ph=False, generators=True, spin=1, double_group=False)
    g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    assert not pg.consistent_U
    pg.fix_U_phases()
    assert pg.consistent_U
    # Phase fixing can multiply the irrep by some 1D rep, result is not always the same
    assert allclose(sum(pg.decompose_U_rep), 1)

    g = cubic(tr=False, ph=False, generators=True, spin=1/2, double_group=True)
    g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    assert not pg.consistent_U
    pg.fix_U_phases()
    assert pg.consistent_U
    # Phase fixing can multiply the irrep by some 1D rep, result is not always the same
    assert allclose(sum(pg.decompose_U_rep), 1)

    g = cubic(tr=False, ph=False, generators=True, spin=1/2, double_group=True)
    g = [PointGroupElement(h.R, U=np.kron(np.eye(2), h.U), RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
