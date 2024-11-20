import pytest
import sympy
import numpy as np
import tinyarray as ta
from itertools import product

from ..groups import PointGroupElement, Model, time_reversal, chiral, rotation, generate_group, cubic
from ..linalg import allclose
from ..characters import SpaceGroupElement, LittleGroupElement, PointGroup, SpaceGroup, LittleGroup, sort_characters


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
    assert allclose((L1**4).phase, 1)
    assert allclose((L1 * (L1**(-1))).phase, 1)
    np.testing.assert_equal(L1 * L1**(-1), L1.identity())
    assert not L1 == L2

    # Test other phase convention
    L1 = LittleGroupElement(S1, k=[0, 0, 1/2], phase=1, phase_in_factor=False)
    L2 = LittleGroupElement(S2, k=[0, 0, 1/2], phase=1, phase_in_factor=False)

    assert allclose((L1 * L2).t, [0, 0, 1/4])
    assert allclose((L1 * L2).phase, 1)
    assert allclose((L1**4).phase, -1)
    assert allclose((L1 * (L1**(-1))).phase, 1)
    np.testing.assert_equal(L1 * L1**(-1), L1.identity())
    assert not L1 == L2

    assert type(S1) == SpaceGroupElement

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
    irrep_dims = [2, 2, 2, 2, 4, 4]
    irreps = pg.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)

def test_little_group_screw():

    def test_screw_reps(n, double_group):
        R1 = rotation(1/n, [0, 0, 1], double_group=double_group)
        if n % 3 == 0:
            periods = np.array([[1, 0, 0], [1/2, np.sqrt(3)/2, 0], [0, 0, 1]])
        else:
            periods = np.eye(3)
        S1 = SpaceGroupElement(R1, t=[0, 0, 1/n], periods=periods)
        SG = SpaceGroup([S1])
        k=[0, 0, 1/2]
        LG = SG.little_group(k=k, phase_in_factor=False)
        # LG._tests = True
        ct = LG.character_table
        gen = LittleGroupElement(S1, k=k)
        ct_ref = np.array([[np.exp(1j * np.pi / n * (1 + 2*i) * j)
                            for j in range(-n//2 + 1, n//2 + 1)]
                           for i in range(n)])
        ordering = np.argsort([LG.class_by_element[gen**i] for i in range(-n//2 + 1, n//2 + 1)])
        ct_ref = ct_ref[:, ordering]
        ct_ref, _ = sort_characters(ct_ref)
        if double_group:
            # double the single group reps
            ct_ref_2 = np.zeros((ct_ref.shape[0], ct_ref.shape[1] * 2), dtype=complex)
            ct_ref_2[:, ::2] = ct_ref
            ct_ref_2[:, 1::2] = ct_ref
            ct_ref = ct_ref_2
        else:
            assert allclose([g.R for g in LG.class_representatives], np.array([(gen**i).R for i in range(-n//2 + 1, n//2 + 1)])[ordering])

        if double_group:
            ### TODO: actually check the double reps, now only checking the single part
            assert allclose(ct[:ct_ref.shape[0]], ct_ref)
        else:
            assert allclose(ct, ct_ref)
        irreps = LG.irreps

        reality = [ir.reality for ir in irreps]
        if not double_group:
            if n == 3:
                assert allclose(reality, [0, 0, 1])
            else:
                assert allclose(reality, 0)
        elif n == 2:
            assert allclose(reality, [0, 0, 1, 1])
        elif n == 3:
            assert allclose(reality, [0, 0, 1, 0, 0, 1])
        elif n == 4:
            assert allclose(reality, [0, 0, 0, 0, 1, 1, 0, 0])
        elif n == 6:
            assert allclose(reality, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])

    for n, double_group in product([2, 3, 4, 6], [True, False]):
        test_screw_reps(n, double_group)

@pytest.mark.parametrize(
    "n, irrep_dims, C2x_column, reality",
    [
        (2, [1, 1, 1, 1, 2], [1j, -1j, -1j, 1j, 0], [0, 0, 0, 0, 1]),
        (3, [1, 1, 1, 1, 2, 2], [-1, 1, 1j, -1j, 0, 0], [1, 1, 0, 0, 1, -1]),
        (4, [1, 1, 1, 1, 2, 2, 2], [1j, -1j, 1j, -1j, 0, 0, 0], [0, 0, 0, 0, 1, 1, -1]),
        (6, [1, 1, 1, 1, 2, 2, 2, 2, 2], [ 1j, -1j, -1j, 1j, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, -1, -1])
    ]
)
def test_screw_C2(n, irrep_dims, C2x_column, reality):
    R1 = rotation(1/n, [0, 0, 1], double_group=True)
    R2 = rotation(1/2, [1, 0, 0], double_group=True)
    if n % 3 == 0:
        periods = np.array([[1, 0, 0], [1/2, np.sqrt(3)/2, 0], [0, 0, 1]])
    else:
        periods = np.eye(3)
    S1 = SpaceGroupElement(R1, t=[0, 0, 1/n], periods=periods)
    S2 = SpaceGroupElement(R2, t=[0, 0, 0], periods=periods)
    SG = SpaceGroup([S1, S2])
    k=[0, 0, 1/2]
    LG = SG.little_group(k=k, phase_in_factor=False)
    # LG._tests = True
    ct = LG.character_table
    assert allclose(LG.character_table[:, 0], irrep_dims)
    assert allclose(LG.character_table_full[:, LG.unitary_elements_list.index(LittleGroupElement(S2, k))], C2x_column)
    irreps = LG.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality for i in irreps], reality)

@pytest.mark.parametrize(
    "k, irrep_dims, C2_column, reality",
    [
        ([0, 0, 0], [1, 1, 1, 2, 2, 2, 3], [1, 1, 1, 0, 0, 0, -1], [1, 0, 0, 0, 0, -1, 1]),
        ([1/2, 1/2, 1/2], [1, 1, 1, 2, 2, 2, 3], [1, 1, 1, 0, 0, 0, -1], [1, 0, 0, 0, 0, -1, 1]),
        ([1/2, 1/2, 0], [1, 1, 1, 1, 2], [1j, 1j, -1j, -1j, 0], [0, 0, 0, 0, 1]),
        ([1/2, 0, 0], [1, 1, 1, 1, 2], [1, 1, -1, -1, 0], [0, 0, 0, 0, 1])
    ]
)
def test_little_group_198(k, irrep_dims, C2_column, reality):
    # SG198
    C2z = rotation(1/2, [0, 0, 1], double_group=True)
    C2z = SpaceGroupElement(C2z, t=[1/2, 0, 1/2], periods=np.eye(3))
    C3 = rotation(1/3, [1, 1, 1], double_group=True)
    C3 = SpaceGroupElement(C3, t=[0, 0, 0], periods=np.eye(3))

    SG = SpaceGroup([C2z, C3])
    assert len(SG.elements) == 24

    LG = SG.little_group(k, phase_in_factor=True)
    # LG._tests = True
    assert len(LG.elements) == sum([d**2 for d in irrep_dims])
    assert allclose(LG.character_table[:, 0], irrep_dims)
    assert allclose(LG.character_table[:, LG.class_by_element[LittleGroupElement(C2z, k)]], C2_column)
    irreps = LG.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
    assert allclose([i.reality for i in irreps], reality)

@pytest.mark.parametrize(
    "k, irrep_dims",
    [
        ([0, 0, 0], [4, 2]),
        ([1/2, 1/2, 1/2], [2, 2, 2, 6]),
        ([1/2, 1/2, 0], [4]),
        ([1/2, 0, 0], [2, 2])
    ]
)
def test_little_group_198_TR(k, irrep_dims):
    # SG198
    C2z = rotation(1/2, [0, 0, 1], double_group=True)
    C2z = SpaceGroupElement(C2z, t=[1/2, 0, 1/2], periods=np.eye(3))
    C3 = rotation(1/3, [1, 1, 1], double_group=True)
    C3 = SpaceGroupElement(C3, t=[0, 0, 0], periods=np.eye(3))
    TR = time_reversal(3, double_group=True)
    TR = SpaceGroupElement(TR, t=[0, 0, 0], periods=np.eye(3))

    SG = SpaceGroup([C2z, C3, TR])
    assert len(SG.elements) == 48

    LG = SG.little_group(k)
    # LG._tests = True
    irreps = LG.irreps
    assert allclose([i.U_shape[0] for i in irreps], irrep_dims)
