import itertools as it
import pytest
import sympy

from .. import kwant_rmt
from ..symmetry_finder import *
from ..symmetry_finder import _reduced_model, _reduce_hamiltonian, bravais_point_group
from ..linalg import *
from .. import kwant_continuum

sigma = np.array([[[1, 0], [0, 1]], [[0, 1], [ 1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

# Stuff for testing
def productrep(n, gens):
    # Make product of n copies of LA rep defined by gens
    genn = gens
    for _ in range(n-1):
        genn = np.array([(np.kron(genn[i],np.eye(len(gens[i])))+
                          np.kron(np.eye(len(genn[i])),gens[i])) for i in range(len(gens))])
    return genn


def sumrep(*args):
    # Make direct sum of n LA reps defined by list of LA's args
    genn = np.array([la.block_diag(*[args[j][i] for j in range(len(args))])
                    for i in range(len(args[0]))])
    return genn


def test_cont_finder():
    # Test symmetry adapted basis
    gens = sumrep((*2*[0.5*sigma[[3, 1, 2]]]))
    U2 = kwant_rmt.circular(len(gens[0]))
    gens2 = np.einsum('ij,ajk,kl->ail',(U2.T).conjugate(),gens,U2)
    U = symmetry_adapted_sun(gens, check=True)
    U = np.hstack(U)
    gensr = np.einsum('ij,ajk,kl->ail',(U.T).conjugate(),gens,U)
    U = symmetry_adapted_sun(gens2, check=True)
    U = np.hstack(U)
    gens2r = np.einsum('ij,ajk,kl->ail',(U.T).conjugate(),gens2,U)
    # check reduced generators are identical
    assert np.allclose(gensr, gens2r), 'Reduced generators should be identical'

    # Test solve_mat_eqn
    n = np.random.randint(2, 5)
    h1 = np.random.random((2, n, n))
    h2 = np.random.random((2, n, n))

    H = np.array([la.block_diag(h1[i], h1[i]) for i in range(len(h1))])
    H = H + (np.einsum('ijk->ikj', H)).conjugate()
    assert solve_mat_eqn(H, H, hermitian=True, traceless=True).shape == (3, 2*n, 2*n)
    assert solve_mat_eqn(H, H, hermitian=False, traceless=True).shape == (3, 2*n, 2*n)
    assert solve_mat_eqn(H, H, hermitian=True, traceless=False).shape == (4, 2*n, 2*n)

    H = np.array([la.block_diag(h1[i], h1[i], h1[i]) for i in range(len(h1))])
    H = H + (np.einsum('ijk->ikj', H)).conjugate()
    assert solve_mat_eqn(H, H, hermitian=True, traceless=True).shape == (8, 3*n, 3*n)
    assert solve_mat_eqn(H, H, hermitian=False, traceless=True).shape == (8, 3*n, 3*n)
    assert solve_mat_eqn(H, H, hermitian=True, traceless=False).shape == (9, 3*n, 3*n)

    H = np.array([la.block_diag(h1[i], h2[i]) for i in range(len(h1))])
    H = H + (np.einsum('ijk->ikj', H)).conjugate()
    assert solve_mat_eqn(H, H, hermitian=True, traceless=True).shape == (1, 2*n, 2*n)
    assert solve_mat_eqn(H, H, hermitian=False, traceless=True).shape == (1, 2*n, 2*n)
    assert solve_mat_eqn(H, H, hermitian=True, traceless=False).shape == (2, 2*n, 2*n)

    # Test _reduce_hamiltonian
    Hs = [h1, np.array([la.block_diag(h1[i], h1[i]) for i in range(len(h1))]),
         np.array([la.block_diag(h1[i], h2[i]) for i in range(len(h1))])]
    for H in Hs:
        H = H + (np.einsum('ijk->ikj', H)).conjugate()
        Ps = _reduce_hamiltonian(H)
        # Check it is proportional to the identity in every block
        for P in Ps:
            Hr = mtm(P[0].T.conjugate(), H, P[0])
            for i, j in it.product(range(len(P)), repeat=2):
                if i ==j:
                    assert np.allclose(mtm(P[i].T.conjugate(), H, P[j]), Hr)
                else:
                    assert np.allclose(mtm(P[i].T.conjugate(), H, P[j]), 0)
        # Check that it vanishes between every block
        for P1, P2 in it.combinations(Ps, 2):
            P1, P2 = np.hstack(P1), np.hstack(P2)
            assert np.allclose(mtm(P1.T.conjugate(), H, P2), 0)


def test_disc_finder(verbose = False):
    """Many test cases for finding discrete onsite symmetries in 0 dimensions"""

    # Thorough testing for all symmetry classes with a single block in a random basis
    sym_list = 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'DIII', 'C', 'CI'
    dim = 8  # Dimension of each block
             # need 8 because must be multiple of 4 for CII and DIII has symmetry with dim<=4
    for sym in sym_list:
        Hs = []
        for _ in range(4):
            h = kwant_rmt.gaussian(dim, sym)
            Hs.append(h)
        Hs = np.array(Hs)
        # Randomize basis
        U = kwant_rmt.circular(dim)
        Hs = np.einsum('ij,ajk,kl->ail',U.T.conj(),Hs,U)
        Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
        # Find symmetry operators
        sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
        sg, Ps = discrete_symmetries(Hs, sg)
        if verbose:
            print('sym', sg)
        T, P, C = kwant_rmt.t(sym), kwant_rmt.p(sym), kwant_rmt.c(sym)
        # check that there are no extra symmetries
        assert len(Ps) == 1
        assert [P.shape for P in Ps] == [(1, dim, dim)]
        assert np.sum(np.abs([T, P, C])) + 1 == len(sg), print([(g.conjugate, g.antisymmetry) for g in sg], T, P, C)
        if T:
            # Check that there is TR
            s = (True, False)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, T*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            assert g.apply(Hs) == Hs
            for H in Hs.values():
                assert np.allclose(H, S.dot(H.conj()).dot(S.T.conj()))
        if P:
            # Check that there is TR
            s = (True, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, P*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H.conj()).dot(S.T.conj()))
        if C:
            # Check that there is TR
            s = (False, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S)
            assert np.allclose(S2, C*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H).dot(S.T.conj()))
    ##############

    # Thorough testing for all symmetry classes with a two different blocks
    # with the same symmetry in a random basis, also tests _reduce_hamiltonian
    sym_list = 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'C', 'CI'
    dim = 4  # Dimension of each block
             # skip DIII because it has symmetry with dim<=4
    for sym in sym_list:
        Hs = []
        for _ in range(4):
            h1 = kwant_rmt.gaussian(dim, sym)
            h2 = kwant_rmt.gaussian(dim, sym)
            Hs.append(la.block_diag(h1, h2))
        Hs = np.array(Hs)
        # Randomize basis
        U = kwant_rmt.circular(2*dim)
        Hs = np.einsum('ij,ajk,kl->ail',U.T.conj(),Hs,U)
        Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
        # Find symmetry operators
        sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
        sg, Ps = discrete_symmetries(Hs, sg)
        if verbose:
            print('sym', sg)
        T, P, C = kwant_rmt.t(sym), kwant_rmt.p(sym), kwant_rmt.c(sym)
        # check that there are exactly two blocks
        assert len(Ps) == 2
        assert [P.shape for P in Ps] == [(1, 2*dim, dim), (1, 2*dim, dim)]
        assert np.sum(np.abs([T, P, C])) + 1 == len(sg), print([(g.conjugate, g.antisymmetry) for g in sg], T, P, C)
        if T:
            # Check that there is TR
            s = (True, False)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, T*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            assert g.apply(Hs) == Hs
            for H in Hs.values():
                assert np.allclose(H, S.dot(H.conj()).dot(S.T.conj()))
        if P:
            # Check that there is TR
            s = (True, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, P*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H.conj()).dot(S.T.conj()))
        if C:
            # Check that there is TR
            s = (False, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S)
            assert np.allclose(S2, C*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H).dot(S.T.conj()))
    ##############

    # Thorough testing for all symmetry classes with a two identical blocks
    # with the same symmetry in a random basis, also tests _reduce_hamiltonian
    # The square of TR and PH may change, only thest 4th power
    sym_list = 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'C', 'CI'
    dim = 4  # Dimension of each block
             # skip DIII because it has symmetry with dim<=4
    for sym in sym_list:
        Hs = []
        for _ in range(4):
            h = kwant_rmt.gaussian(dim, sym)
            Hs.append(la.block_diag(h, h))
        Hs = np.array(Hs)
        # Randomize basis
        U = kwant_rmt.circular(2*dim)
        Hs = np.einsum('ij,ajk,kl->ail',U.T.conj(),Hs,U)
        Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
        # Find symmetry operators
        sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
        sg, Ps = discrete_symmetries(Hs, sg)
        if verbose:
            print('sym', sg)
        T, P, C = kwant_rmt.t(sym), kwant_rmt.p(sym), kwant_rmt.c(sym)
        # check that there are exactly two blocks
        assert len(Ps) == 1
        assert [P.shape for P in Ps] == [(2, 2*dim, dim)], (T, P, C)
        assert np.sum(np.abs([T, P, C])) + 1 == len(sg), print([(g.conjugate, g.antisymmetry) for g in sg], T, P, C)
        if T:
            # Check that there is TR
            s = (True, False)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, T*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            assert g.apply(Hs) == Hs
            for H in Hs.values():
                assert np.allclose(H, S.dot(H.conj()).dot(S.T.conj()))
        if P:
            # Check that there is TR
            s = (True, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S.conj())
            assert np.allclose(S2, P*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H.conj()).dot(S.T.conj()))
        if C:
            # Check that there is TR
            s = (False, True)
            g = [g for g in sg if (g.conjugate, g.antisymmetry) == s]
            assert len(g) == 1
            g = g[0]
            # Check square
            S = g.U
            S2 = S.dot(S)
            assert np.allclose(S2, C*np.eye(S2.shape[0]))
            # Check that it is a symmetry
            for H in Hs.values():
                assert np.allclose(-H, S.dot(H).dot(S.T.conj()))
    ##############

    # Two blocks related by time reversal symmetry
    dim = 4  # Dimension of each block
    sym = 'AII'  # Time-reversal squares to -1
    # Time reversal operator for each block
    t_mat = np.array(kwant_rmt.h_t_matrix[sym])
    t_mat = np.kron(np.identity(dim // len(t_mat)), t_mat)
    # Hamiltonians
    # Consider a family of Hamiltonians with two blocks related by time-reversal
    Hs = []
    for _ in range(4):
        h = kwant_rmt.gaussian(dim) # Random Hamiltonian with no symmetry
        # Blocks related by time-reversal
        Hs.append(la.block_diag(h, t_mat.dot(h.conj()).dot(t_mat.T.conj())))
    # Complete time-reversal operator
    sx = np.array([[0, 1], [1, 0]])
    T = np.kron(sx, t_mat)
    for H in Hs:
        assert np.allclose(H, T.dot(H.conj()).dot(T.T.conj()))
    # Find the symmetry operator
    Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
    sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
    sg, Ps = discrete_symmetries(Hs, sg)
    assert [P.shape for P in Ps] == [(1, 2*dim, dim), (1, 2*dim, dim)]
    assert len(sg) == 2
    tr = list(sg)[0]
    TR = tr.U
    # Squares to +1
    assert np.allclose(TR.dot(TR.conj()), np.eye(TR.shape[0]))

    if verbose:
        print('TR**2', TR.dot(TR.conj()))
    assert np.allclose(TR.dot(TR.conj()).dot(TR.dot(TR.conj())), np.eye(TR.shape[0]))
    # Is a symmetry of all Hamiltonians
    for H in Hs:
        assert tr.apply(Hs) == Hs
    ##########################

    # Add a third block, one which has time-reversal symmetry by itself, but is not related to the other two.
    dim = 4  # Dimension of each block
    sym = 'AII'  # Time-reversal squares to -1
    # Time reversal operator for each block
    t_mat = np.array(kwant_rmt.h_t_matrix[sym])
    t_mat = np.kron(np.identity(dim // len(t_mat)), t_mat)

    # Consider a family of Hamiltonians with two blocks related by time-reversal,
    # neither of which has time-reversal, and a third block that has time-reversal.
    Hs = []
    for _ in range(8):
        h = kwant_rmt.gaussian(dim) # Random Hamiltonian with no symmetry
        ht = kwant_rmt.gaussian(dim, sym)
        Hs.append(la.block_diag(ht, h, t_mat.dot(h.conj()).dot(t_mat.T.conj())))
    # Complete time-reversal operator
    sx = np.array([[0, 1], [1, 0]])
    T = la.block_diag(t_mat, np.kron(sx, t_mat))
    for H in Hs:
        assert np.allclose(H, T.dot(H.conj()).dot(T.T.conj()))
    Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
    sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
    sg, Ps = discrete_symmetries(Hs, sg)
    assert [P.shape for P in Ps] == [(1, 3*dim, dim), (1, 3*dim, dim), (1, 3*dim, dim)]
    assert len(sg) == 2
    tr = list(sg)[0]
    TR = tr.U
    if verbose:
        print('TR**2', TR.dot(TR.conj()))
    assert np.allclose(TR.dot(TR.conj()).dot(TR.dot(TR.conj())), np.eye(TR.shape[0]))
    # Is a symmetry of all Hamiltonians
    for H in Hs:
        assert tr.apply(Hs) == Hs
    ##########################

    # Two blocks related by time reversal symmetry, with each block having
    # time-reversal symmetry. Here, there are two ways to consruct the
    # time-reversal operator and there is a unitary symmetry relating the blocks
    dim = 4  # Dimension of each block
    sym = 'AII'  # Time-reversal squares to -1
    # Time reversal operator for each block
    t_mat = np.array(kwant_rmt.h_t_matrix[sym])
    t_mat = np.kron(np.identity(dim // len(t_mat)), t_mat)
    # Hamiltonians
    # Consider a family of Hamiltonians with two blocks related by time-reversal
    Hs = []
    for _ in range(8):
        h = kwant_rmt.gaussian(dim, sym) # Random Hamiltonian with time-reversal
        # Blocks related by time-reversal, each with time-reversal
        Hs.append(la.block_diag(h, t_mat.dot(h.conj()).dot(t_mat.T.conj())))
    # Complete time-reversal operator
    sx = np.array([[0, 1], [1, 0]])
    T = np.kron(sx, t_mat)
    for H in Hs:
        assert np.allclose(H, T.dot(H.conj()).dot(T.T.conj()))
    Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
    sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
    sg, Ps = discrete_symmetries(Hs, sg)
    assert [P.shape for P in Ps] == [(2, 2*dim, dim)]
    assert len(sg) == 2
    tr = list(sg)[0]
    TR = tr.U
    if verbose:
        print('TR**2', TR.dot(TR.conj()))
    assert (np.allclose(TR.dot(TR.conj()), np.eye(TR.shape[0])) or
            np.allclose(TR.dot(TR.conj()), -np.eye(TR.shape[0])))
    # Is a symmetry of all Hamiltonians
    for H in Hs:
        assert tr.apply(Hs) == Hs
    ############

    # Four blocks. The first two related by time-reversal but neither
    # possessing it themselves. The second two both possess time-reversal
    # and are related by it.
    dim = 4  # Dimension of each block
    sym = 'AII'  # Time-reversal squares to -1
    # Time reversal operator for each block
    t_mat = np.array(kwant_rmt.h_t_matrix[sym])
    t_mat = np.kron(np.identity(dim // len(t_mat)), t_mat)

    # Hamiltonians
    # Consider a family of Hamiltonians with two blocks related by time-reversal
    Hs = []
    for _ in range(8):
        h1 = kwant_rmt.gaussian(dim, sym) # Random Hamiltonian with time-reversal
        h2 = kwant_rmt.gaussian(dim) # Random Hamiltonian with no symmetry
        Hs.append(la.block_diag(h2, t_mat.dot(h2.conj()).dot(t_mat.T.conj()),
                                h1, t_mat.dot(h1.conj()).dot(t_mat.T.conj())))
    Hs = Model({kwant_continuum.sympify('a_' + str(i)) : H for i, H in enumerate(Hs)},
                           momenta=[])
    sg = {PointGroupElement(np.empty((0, 0)), c, a)
              for c, a in [[True, False], [True, True], [False, True]]}
    sg, Ps = discrete_symmetries(Hs, sg)
    assert sorted([P.shape for P in Ps]) == sorted([(1, 4*dim, dim), (1, 4*dim, dim), (2, 4*dim, dim)])
    assert len(sg) == 2
    tr = list(sg)[0]
    TR = tr.U
    if verbose:
        print('TR**2', TR.dot(TR.conj()))
    assert np.allclose(TR.dot(TR.conj()).dot(TR.dot(TR.conj())), np.eye(TR.shape[0]))
    # Is a symmetry of all Hamiltonians
    for H in Hs:
        assert tr.apply(Hs) == Hs
    ##########################

def test_simult_diag():
    # Make diagonal unitary and hermitian matrices with degenerate eigenvalues
    Hs = [np.diag(np.exp(1j * np.random.randint(-3, 3, (20)))) for _ in range(3)]
    Hs += [np.diag(np.random.randint(-3, 3, (20))) for _ in range(3)]
    # random unitary rotation
    U = kwant_rmt.circular(len(Hs[0]))
    Hs = U @ Hs @ U.T.conj()
    # undo the rotation with simult_diag
    U = np.hstack(simult_diag(Hs))
    Hsd = (U.T.conj() @ Hs @ U)
    # result is diagonal
    assert np.allclose(np.abs([H - np.diag(H.diagonal()) for H in Hsd]), 0)

def test_continuum():
    # Simple tests for continuum models

    # Cubic point group
    eye = np.array(np.eye(3), int)
    E = PointGroupElement(eye, False, False, None)
    I = PointGroupElement(-eye, False, False, None)
    C4 = PointGroupElement(np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]], int), False, False, None)
    C3 = PointGroupElement(np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0, 1, 0]], int), False, False, None)
    TR = PointGroupElement(eye, True, False, None)
    PH = PointGroupElement(eye, True, True, None)
    cubic_gens = {I, C4, C3, TR, PH}
    cubic_group = generate_group(cubic_gens)
    assert len(cubic_group) == 192

    # First model
    ham1 = ("hbar^2 / (2 * m) * (k_x**2 + k_y**2 + k_z**2) * eye(2) +" +
        "alpha * sigma_x * k_x + alpha * sigma_y * k_y + alpha * sigma_z * k_z")
    # Convert to standard model form
    H1 = Model(ham1)
    sg, Ps = discrete_symmetries(H1, cubic_group)
    assert [P.shape for P in Ps] == [(1, 2, 2)]
    assert len(sg) == 48
    assert sg == generate_group({C4, C3, TR})

    # Add a degeneracy
    ham2 = "kron(eye(2), " + ham1 + ")"
    # Convert to standard model form
    H2 = Model(ham2)
    sg, Ps = discrete_symmetries(H2, cubic_group)
    assert [P.shape for P in Ps] == [(2, 4, 2)]
    assert len(sg) == 48
    assert sg == generate_group({C4, C3, TR})

    # Add hole degrees of freedom
    ham2 = "kron(sigma_z, " + ham1 + ")"
    # Convert to standard model form
    H3 = Model(ham2)
    sg, Ps = discrete_symmetries(H3, cubic_group)
    assert [P.shape for P in Ps] == [(1, 4, 2), (1, 4, 2)]
    assert len(sg) == 96
    assert sg == generate_group({C4, C3, TR, PH})

    # Continuous rotation symmetry
    for H in [H1, H2, H3]:
        assert len(continuous_symmetries(H)) == 3

def test_bloch():
    # Simple tests for Bloch models

    # Hexagonal point group
    eyesym = sympy.ImmutableMatrix(sympy.eye(2))
    Mx = PointGroupElement(sympy.ImmutableMatrix([[-1, 0],
                                                  [0, 1]]),
                            False, False, None)
    C6 = PointGroupElement(sympy.ImmutableMatrix([[sympy.Rational(1, 2), sympy.sqrt(3)/2],
                                                  [-sympy.sqrt(3)/2,       sympy.Rational(1, 2)]]),
                                 False, False, None)
    TR = PointGroupElement(eyesym, True, False, None)
    PH = PointGroupElement(eyesym, True, True, None)
    gens_hex_2D ={Mx, C6, TR, PH}
    hex_group_2D = generate_group(gens_hex_2D)
    assert len(hex_group_2D) == 48

    # First example
    ham6 = 'm * (cos(k_x) + cos(1/2*k_x + sqrt(3)/2*k_y) + cos(-1/2*k_x + sqrt(3)/2*k_y))'
    H6 = Model(ham6, momenta=[0, 1])
    sg, Ps = discrete_symmetries(H6, hex_group_2D)
    assert [P.shape for P in Ps] == [(1, 1, 1)]
    assert len(sg) == 24
    assert sg == generate_group({Mx, C6, TR})

    # extend model to add SOC
    ham62 = 'eye(2) * (' + ham6 + ') +'
    ham62 += 'alpha * (sin(k_x) * sigma_x + sin(1/2*k_x + sqrt(3)/2*k_y) * (1/2*sigma_x + sqrt(3)/2*sigma_y) +'
    ham62 += 'sin(-1/2*k_x + sqrt(3)/2*k_y) * (-1/2*sigma_x + sqrt(3)/2*sigma_y))'
    H62 = Model(ham62, momenta=[0, 1])
    sg, Ps = discrete_symmetries(H62, hex_group_2D)
    assert [P.shape for P in Ps] == [(1, 2, 2)]
    assert len(sg) == 24
    assert sg == generate_group({Mx, C6, TR})

    # Add degeneracy
    ham63 = 'kron(eye(2), ' + ham62 + ')'
    H63 = Model(ham63, momenta=[0, 1])
    sg, Ps = discrete_symmetries(H63, hex_group_2D)
    assert [P.shape for P in Ps] == [(2, 4, 2)]
    assert len(sg) == 24
    assert sg == generate_group({Mx, C6, TR})

    # Add PH states
    ham64 = 'kron(sigma_z, ' + ham62 + ')'
    H64 = Model(ham64, momenta=[0, 1])
    sg, Ps = discrete_symmetries(H64, hex_group_2D)
    assert [P.shape for P in Ps] == [(1, 4, 2), (1, 4, 2)]
    assert len(sg) == 48
    assert sg == generate_group({Mx, C6, TR, PH})


def test_bravais_symmetry():
    # 2D
    # random rotation
    R = kwant_rmt.circular(2, sym='D', rng=1)
    lattices = [
                ([[np.sqrt(3)/2, 1/2], [0, 1]], 12, 'hexagonal'),
                ([[1, 0], [0, 1]], 8, 'square'),
                ([[-1/3, 1], [1/3, 1]], 4, 'centered orthorhombic')
                ]
    for periods, n, name in lattices:
        group = bravais_point_group(periods, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)
        group = bravais_point_group(periods @ R, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)

    # 3D
    # random rotation
    R = kwant_rmt.circular(3, sym='D', rng=1)
    lattices = [
                (np.eye(3), 48, 'primitive cubic'),
                ([[1, 0, 0], [0, 1, 0], [1/2, 1/2, 1/2]], 48, 'BCC'),
                ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], 48, 'FCC'),
                ([[1, 0, 0], [0, 1, 0], [0, 0, 3]], 16, 'primitive tetragonal'),
                ([[1, 0, 0], [0, 1, 0], [1/2, 1/2, 3]], 16, 'body centered tetragonal'),
                ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], 8, 'primitive orthorhombic'),
                ([[1, 0, 0], [0, 2, 0], [1/2, 1, 3]], 8, 'body centered orthorhombic'),
                ([[10, 3, 0], [10, 0, 4], [0, 3, 4]], 8, 'face centered orthorhombic'),
                ([[1, 3, 0], [1, 0, 4], [0, 3, 4]], 8, 'face centered orthorhombic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [0, 0, 4]], 8, 'base centered orthorhombic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [0, 0, np.sqrt(10)/3]], 8, 'base centered orthorhombic corner case'),
                ([[1, 0, 0], [0, 2, 0], [1/10, 0, 4]], 4, 'primitive monoclinic'),
                ([[1, 0, 0], [0, 1, 0], [0, 1/10, 2]], 4, 'primitive monoclinic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [1/10, 0, 4]], 4, 'base centered monoclinic'),
                ([[3, 0, 1], [3/2, 1, 1/2], [0, 0, 4]], 4, 'base centered monoclinic'),
                ([[3, 0, 1], [3/2, 1/2, 1/2], [0, 0, 4]], 4, 'base centered monoclinic'),
                ([[1, 0, 1/10], [0, 2, 1/10], [0, 0, 3]], 2, 'triclinic'),
                ([[1, 0, 1/5], [-1/2, np.sqrt(3)/2, 1/5], [-1/2, -np.sqrt(3)/2, 1/5]], 12, 'rhombohedral'),
                ([[1, 0, 5], [-1/2, np.sqrt(3)/2, 5], [-1/2, -np.sqrt(3)/2, 5]], 12, 'rhombohedral'),
                ([[np.sqrt(3)/2, 1/2, 0], [0, 1, 0], [0, 0, 2]], 24, 'hexagonal'),
                ([[np.sqrt(3)/2, 1/2, 0], [0, 1, 0], [0, 0, 1]], 24, 'hexagonal equal length corner case'),
                ]
    for periods, n, name in lattices:
        group = bravais_point_group(periods, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)
        group = bravais_point_group(periods @ R, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)
