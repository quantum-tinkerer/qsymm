import pytest
import numpy as np
import tinyarray as ta

from ..groups import PointGroupElement, PointGroup, time_reversal, rotation, cubic
from ..linalg import allclose


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

def test_pointgroup_cubic():
    # turn on _tests for deep self-testing, takes longer
    pg = PointGroup(cubic(tr=False, ph=False, generators=True, double_group=False))
    pg._tests=True
    irrep_dims = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    assert allclose(pg.character_table[:, 0], irrep_dims)
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality for i in irreps], np.ones((10,)))

    pg = PointGroup(cubic(tr=False, ph=False, generators=True, double_group=True))
    # pg._tests=True
    irrep_dims = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    assert allclose(pg.character_table[:, 0], irrep_dims)
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality for i in irreps], [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1])

    # Test phase fixing
    g = cubic(tr=False, ph=False, generators=True, spin=1, double_group=False)
    g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    # pg._tests=True
    assert not pg.consistent_U
    pg.fix_U_phases()
    assert pg.consistent_U
    # Phase fixing can multiply the irrep by some 1D rep, result is not always the same
    assert allclose(sum(pg.decompose_U_rep), 1)

    g = cubic(tr=False, ph=False, generators=True, spin=1/2, double_group=True)
    g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    # pg._tests=True
    assert not pg.consistent_U
    pg.fix_U_phases()
    assert pg.consistent_U
    # Phase fixing can multiply the irrep by some 1D rep, result is not always the same
    assert allclose(sum(pg.decompose_U_rep[4:10]), 1)

    g = cubic(tr=False, ph=False, generators=True, spin=1/2, double_group=True)
    g = [PointGroupElement(h.R, U=np.kron(np.eye(2), h.U), RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    # pg._tests=True
    assert allclose(pg.decompose_U_rep, [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])

    # Test symmetry_adapted_basis
    n = 3
    spin = 1/2
    d = int((2 * spin + 1) * n)
    g = cubic(tr=False, ph=False, generators=True, spin=spin, double_group=True)
    g = [PointGroupElement(h.R, U=np.kron(np.eye(n), h.U), RSU2=h.RSU2) for h in g]
    ### TODO: use reproducible pseudorandom numbers
    W, _, _ = np.linalg.svd(np.random.normal(size=(d, d)) + 1j * np.random.normal(size=(d, d)))
    g = [PointGroupElement(h.R, U=W @ h.U @ W.conj().T, RSU2=h.RSU2) for h in g]
    pg = PointGroup(g)
    # pg._tests=True
    assert allclose(pg.decompose_U_rep, [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0])
    sb = pg.symmetry_adapted_basis
    bb = np.hstack(sb)
    gen_tr = np.array([sb[0].T.conj() @ gen.U @ sb[0] for gen in pg.minimal_generators])
    for U in sb[1:]:
        assert allclose([U.T.conj() @ gen.U @ U for gen in pg.minimal_generators], gen_tr)

def test_pointgroup_permutation():

    def permutation_generators(n):
        gens = []
        for i in range(n-1):
            g = np.eye(n, dtype=int)
            g[0, 0] = g[i+1, i+1] = 0
            g[0, i+1] = g[i+1, 0] = 1
            gens.append(g)
        return np.array(gens)

    def kron_power(a, p):
        b = a
        for i in range(p-1):
            b = np.kron(a, b)
        return b

    def power_rep(gens, p, sparse=False):
        if sparse:
            return [scsp.csr_matrix(kron_power(g, p)) for g in gens]
        else:
            return np.array([kron_power(g, p) for g in gens])

    n = 4
    gens = [PointGroupElement(p) for p in permutation_generators(n)]
    group = PointGroup(gens)
    irrep_dims = [1, 1, 2, 3, 3]
    assert allclose(group.character_table[:, 0], irrep_dims)
    assert allclose(group.decompose_R_rep, [1, 0, 0, 0, 1])

    n = 4
    p = 2
    p_gens = permutation_generators(n)
    reps = power_rep(p_gens, p, sparse=False)
    gens = [PointGroupElement(p, U=U) for p, U in zip(p_gens, reps)]
    group = PointGroup(gens)
    assert allclose(group.decompose_U_rep, [2, 0, 1, 1, 3])

def test_pointgroup_TR():
    pg = PointGroup(cubic(tr=True, ph=False, generators=True, double_group=False))
    irrep_dims = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

    pg = PointGroup(cubic(tr=True, ph=False, generators=True, double_group=True))
    irrep_dims = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

    pg = PointGroup(cubic(tr=True, ph=False, generators=True, double_group=True), double_group='forced')
    irrep_dims = [2, 2, 2, 2, 4, 4]
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

def test_C4T():
    # Test group with TR with nontrivial square
    C4 = rotation(1/4, double_group=False)
    T = time_reversal(2, double_group=False)
    C4T = C4 * T
    pg = PointGroup([C4T])
    irreps = pg.irreps
    irrep_dims = [1, 2]
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

    C4 = rotation(1/4, double_group=True)
    T = time_reversal(2, double_group=True)
    C4T = C4 * T
    pg = PointGroup([C4T], double_group=True)
    irreps = pg.irreps
    irrep_dims = [1, 2, 2]
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

    pg = PointGroup([C4T], double_group='forced')
    irreps = pg.irreps
    irrep_dims = [2]
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
