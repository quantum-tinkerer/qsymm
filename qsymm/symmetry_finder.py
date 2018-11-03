import numpy as np
import scipy.linalg as la
import itertools as it
from copy import deepcopy

from .linalg import matrix_basis, nullspace, split_list, simult_diag, commutator, \
                    prop_to_id, sparse_basis, mtm, family_to_vectors, solve_mat_eqn, \
                    allclose
from .model import Model
from .groups import PointGroupElement, ContinuousGroupGenerator, generate_group, \
                    set_multiply

### Top level function for symmetry finding

def symmetries(model, candidates=None, continuous_rotations=False,
               generators=False, prettify=False, num_digits=8, verbose=False,
               sparse_linalg=False):
    """
    Find symmetries of the Hamiltonian family described by model.

    Parameters:
    -----------
    model : Model
        Model which represents family of Hamiltonians. Every symbolic
        prefactor is treated as a free parameter, and model.momenta
        as independent momentum variables.
    candidates : iterable of PointGroupElements or None
        Set of candidate PointGroupElements used for finding discrete
        symmetries. Must have .U attribute set to None. If model
        describes a Bloch Hamiltonian, the rotation matrices must be
        either integer or Sympy matrices, as exact arythmetic is assumed.
        If None (default), only discrete onsite symmetries (time reversal,
        particle-hole, chiral) are found.
    continuous_rotations : bool
        Whether to search for continuous rotation symmetries.
    generators : bool
        If True, only a set of generators are returned, otherwise
        the full discrete symmetry group is returned.
    prettify : bool
        Whether to carry out sparsification of the continuous symmetry
        generators, in general an arbitrary linear combination of the
        symmetry generators is returned.
    num_digits : float
        Absolute precision when deciding whether symmetry leaves Hamiltonian
        invariant and prettifying the result with prettify=True.
    verbose : bool
        Whether to print additional information.
    sparse_linalg : bool
        Whether to use sparse linear algebra in the calculation.
        Can give large performance gain in large systems.

    Returns:
    --------
    disc_sym : list of PointGroupElement
        Discrete symmetries of model. If generators=False, it is a closed group,
        if generators=True it is a generator set, may be empty.
    cont_sym : list of ContinuousGroupGenerator
        List of linearly independent continuous symmetry generators, may be
        empty. Onsite conserved quantities are listed first, then continuous
        rotation generators. The trivial conserved quantity proportional to
        identity is not included.
    """

    # Find onsite conserved quantites and projectors onto blocks.
    Ps = _reduce_hamiltonian(np.array(list(model.values())), sparse=sparse_linalg)
    cont_sym = conserved_quantities(Ps, prettify=prettify, num_digits=num_digits)

    # Find continuous rotations
    if continuous_rotations:
        cont_sym += continuous_symmetries(model, Ps=Ps, prettify=prettify,
                                          num_digits=num_digits, sparse_linalg=sparse_linalg)

    # Find discrete symmetries
    if candidates is None:
        # Make discrete onsite symmetries
        dim = len(model.momenta)
        candidates = generate_group({PointGroupElement(np.eye(dim), True, False),
                                     PointGroupElement(np.eye(dim), False, True)})

    if candidates:
        disc_sym, _ = discrete_symmetries(model, set(candidates), Ps=Ps,
                                          generators=generators, verbose=verbose,
                                          sparse_linalg=sparse_linalg)
        disc_sym = sorted(list(disc_sym))
    else:
        disc_sym = []

    return disc_sym, cont_sym


### Lie Algebra utility functions

def struct_const(gens):
    # Calculate structure constants of Lie algebra given by 'gens'

    def expand(v, bas):
        # expand v in incomplete nonorthogonal basis bas
        c, res, _, _ = la.lstsq(bas, v)
        if not np.all(np.isclose(res, 0)):
            raise ValueError('Vector outside of space spanned by basis!')
        return c

    bas = np.array([L.flatten() for L in gens]).T
    return np.array(
        [
            [
                expand(commutator(L1,L2).flatten(), bas)
            for L2 in gens]
        for L1 in gens])


def killing(gens):
    # Calculate Killing form
    C = struct_const(gens)
    # 'ade,bed->ab'
    return np.tensordot(C, C, axes=((1, 2), (2, 1)))


def casimir(gens):
    # Calculate quadratic Casimir operator
    g = killing(gens)
    if np.isclose(la.det(g), 0):
        raise ValueError('Lie-algebra not semisimple!')
    # 'ab,aik,bkj->ij'
    return np.tensordot(la.inv(g), gens.dot(gens), axes=((0, 1), (0, 2)))
    # Generalized to the case when includes a commutative subalgebra ???
    # return np.einsum('ab,aik,bkj->ij',la.pinv(g),gens,gens)


def separate_lie_algebra(gens):
    """
    Separate Lie-algebra defined by the list of linearly
    independent generators 'gens' into its center and semisimple parts.

    Parameters:
    -----------------
    gens : ndarray
        3 index array of shape (m, N, N), Lie-algebra generators

    Returns:
    ---------------
    gensc : ndarray
        generators of the center, shape (mc, N, N)
    genss : ndarray
        generators of semisimple part, shape (ms, N, N)
    """
    g = killing(gens)
    c, s = nullspace(g, return_complement=True)
    # 'ab,bij->aij'
    gensc = np.tensordot(c, gens, axes=(1, 0))
    genss = np.tensordot(s, gens, axes=(1, 0))
    return gensc, genss


def symmetry_adapted_sun(gens, check=False):
    """
    Find symmetry adapted basis of the simple 'su(d)'
    Lie-algebra representation defined by generators 'gens'.
    It is assumed that the representation is the direct sum
    of 'n' identical 'su(d)' representations, i.e. in the symmetry
    adapted basis the generators have the form 'L \otimes 1_{nxn}' where
    'L' runs over a generator set of all 'd*d' Hermitian matrices.

    Parameters:
    -----------------
    gens : ndarray
        3 index array of shape '(d**2-1, n*d, n*d)' with 'n' and 'd' integers,
        list of 'su(d)' generators.

    Returns:
    ---------------
    Ps : ndarray
        3 index array of shape '(d, n*d, n)', list of projectors onto the symmetry
        adapted subspaces. 'np.einsum('aij,ab,bkj->ik', Ps, L, Ps.conjugate()))'
        is a symmetry spanned by 'gens' for any 'd*d' Hermitian matrix 'L'.
        Stacking these produces the unitary transformation 'U = np.hstack(Ps)' to the
        symmetry adapted basis.
    check : bool
        Whether to check the final result.
    """
    d = np.sqrt(len(gens) + 1)
    n = gens.shape[-1] / d
    if not (n.is_integer() and d.is_integer()):
        raise ValueError('Shape of gens is incompatible with it being direct sum of'
                         'n identical copies of su(d) representation.')
    n, d = int(n), int(d)
    # Trivial case when it is a full SU(N)
    if n == 1:
        return np.array(np.split(np.eye(n*d), n*d, 1))

    # Iteratively split the irreps. Goal is to find a basis in the n*d dimensional
    # Hilbert-space, where the n vectors belonging to the identity factor in the symmetry
    # adapted basis are grouped together. A degenerate eigensubspace of a generator has
    # dimension of a multiple of n. If it is exactly n, we can be sure that we found
    # such a subspace. If it is f*n, it is the union of f such subspaces. We project the
    # remaining generators in this subspace, they will still span all matrices with the
    # structure L \otimes 1_{nxn} with L f*f. We pick a new generator and repeat the process.
    # In the generic case (a random combination of the generators) the first generator
    # already splits all the irreps, but in the standard matrix basis this is not the case.
    unsplit = [np.eye(n*d)]
    split = []
    for g in gens:
        Ps = unsplit
        unsplit = []
        for P in Ps:
            # project generator in unsplit subspace, rest of the generators still
            # span the full space of Hermitian matrices in the restricted space
            gr = P.T.conj() @ g @ P
            evals, U = la.eigh(gr)
            # find the degenerate eigenspaces
            blocks = split_list(evals)
            for b, e in blocks:
                if e - b == n:
                    # if n long degenerate block, it is split
                    split.append(P @ U[:, b:e])
                else:
                    assert (e - b) % n == 0
                    # if it is a multiple of n, put it back in unsplit
                    unsplit.append(P @ U[:, b:e])
        if unsplit == []:
            break
    else:
        # we could not split all the subspaces
        raise ValueError('Algorithm failed, likely not and su(d) representation')
    Ps = np.array(split)

    P0 = Ps[0]
    for i, P in enumerate(Ps[1:]):
        # Adjust the basis of every block to the basis of the first block,
        # such that all the generators are proportional to the identity
        # in every block.
        for g in gens:
            # find a generator where the block is nonzero
            gblock = P0.T.conj() @ g @ P
            if np.allclose(gblock, 0):
                continue
            # then it is proportional to unitary, transform it
            # to proportional to identity
            u, s, vh = la.svd(gblock)
            U = (((u @ vh).T).conj())
            Ps[i+1] = Ps[i+1] @ U
            break
        else:
            # we could not fix some of the bases
            raise ValueError('Algorithm failed, likely not and su(d) representation')
    if check:
        # check that every block is proportional to the identity
        U = np.hstack(Ps)
        genst = mtm(U.T.conjugate(), gens, U)
        assert np.all([[prop_to_id(genst[i, n*j1:n*(j1+1), n*j2:n*(j2+1)])[0]
                     for j1, j2 in it.combinations_with_replacement(range(d), 2)]
                     for i in range(len(genst))])
    return Ps


### Continuous onsite symmetry finding

def _reduce_hamiltonian(H, sparse=False):
    """
    Find the unitary symmetries and the symmetry adapted basis of a family
    of matrices H. In the symmetry adapted basis the matrices are
    simultaneously block-diagonalized, each block corresponding to an
    irreducible subspace.

    Parameters:
    -----------------
    H : ndarray
        3 index array of shape (m, N, N), list of NxN Hermitian matrices
        defining the family.
    sparse : bool
        Whether to use sparse linear algebra in the calculation.
        Can give large performance gain in large systems.

    Returns:
    ---------------
    Preduced : tuple of ndarrays
        symmetry adapted bases (Pr_1, Pr_2, ...) where Pr_i is a 3 index array of
        shape (d_i, N, n_i) with sum_i n_i * d_i = N, each have the same format
        as the output of symmetry_adapted_sun and satisfies that
        `np.einsum('aij,ab,bkj->ik', Pr_i, L, Pr_i.conjugate()))` is a symmetry
        in the i'th block for any d_i x d_i Hermitian matrix `L`. Each Pr_i
        corresponds to d_i irreducible subspaces of n_i dimensions, in which
        the H's act identically.
    """
    # Find all symmetry generators
    gens = solve_mat_eqn(H, hermitian=True, traceless=True, sparse=sparse)

    if len(gens) == 0:
        return (np.array([np.eye(H.shape[-1])]),)
    # Find the center
    gensc, _ = separate_lie_algebra(gens)
    if len(gensc) == 0:
        Ps = [np.eye(H.shape[-1])]
    else:
        # Simultaneaously diagonalize central generators
        Ps = simult_diag(gensc)

    Preduced = tuple()
    # Find symmetry adapted basis of restricted LA in each subspace
    for P in Ps:
        Hr = mtm(P.T.conjugate(), H, P)
        gensr = solve_mat_eqn(Hr, hermitian=True, traceless=True, sparse=sparse)
        # Find symmetry adapted basis in subspace
        if len(gensr) == 0:
            P2s = [np.eye(Hr.shape[-1])]
        else:
            P2s = symmetry_adapted_sun(gensr)
        Preduced += (np.array([np.dot(P, P2i) for P2i in P2s]),)

    return Preduced


def conserved_quantities(Ps, prettify=False, num_digits=3):
    """
    Construct a full set of conserved quantities from the projectors.

    Parameters:
    -----------
    Ps : list fo 3 index ndarrays
        projectors 'Ps' returned '_reduce_hamiltonian()'
    prettify : bool
        If true it finds a nice sparse basis of the conserved
        quantities, they are generally returned in a random basis, but
        any linear combination is also conserved.

    Returns:
    --------
    list of ContinuousGroupGenerators
        conserved quantities that all commute with the family of
        Hamiltonians. The identity matrix is excluded.
    """
    Ls = []
    # Iterate over symmetry blocks
    for i, P in enumerate(Ps):
        # generate basis of Hermitian matrices with subblock size
        # First block does not have identity, so the identity is not
        # among the conserved quantities
        bas = matrix_basis(P.shape[0], traceless=(i==0))
        for l in bas:
            # construct conserved L that acts with l on all the subblocks
            # of the original space at the same time (sum over j)
            # 'aij,ab,bkj->ik'
            L = np.tensordot(np.tensordot(P, l, axes=((0), (0))), P.conj(),
                             axes=((2, 1), (0, 2)))
            # Make it traceless
            Ls.append(L - np.trace(L) / len(L) * np.eye(len(L)))
    Ls = np.array(Ls)
    # Sparsify the matrices using reduced row echelon form
    if len(Ls) > 1 and prettify:
        Lsf = Ls.reshape(Ls.shape[0], -1)
        Lsf = sparse_basis(Lsf, reals=True, num_digits=num_digits)
        Ls = Lsf.reshape(Lsf.shape[0], *Ls.shape[1:])
    return [ContinuousGroupGenerator(None, L) for L in Ls]


### Point group symmetry finding

def discrete_symmetries(model, candidates, Ps=None, generators=False,
                        verbose=False, sparse_linalg=False):
    """Find point group symmetries of Hamiltonians family.
    Optimized version to reduce number of tests,
    uses sympy exact rotation matrices

    Parameters:
    -----------
    model : Model
        Model which represents family of Hamiltonians
    candidates : set of PointGroupElements
        Set of candidate PointGroupElements. Must have .U attribute
        set to None.
    Ps : ndarray, optional
        Projectors as returned by _reduce_hamiltonian.
    generators : bool
        If true, only a set of generators are returned, otherwise
        the full symmetry group is returned.
    sparse_linalg : bool
        Whether to use sparse linear algebra in the calculation.
        Can give large performance gain in large systems.
    verbose : bool

    Returns:
    --------
    genset or symset : set of PointGroupElement
        Symmetries of model.
    ### TODO: remove Ps from return
    Ps : ndarray
        Projectors as returned by _reduce_hamiltonian.
    """
    symmetry_candidates = deepcopy(candidates)
    m = len(symmetry_candidates)
    # Reduce Hamiltonian by onsite unitaries
    if not Ps:
        Ps = _reduce_hamiltonian(np.array(list(model.values())), sparse=sparse_linalg)
    # After every step, symlist is guaranteed to form a group, start with the trivial group
    e = next(iter(candidates)).identity()
    e.U = np.eye(Ps[0].shape[1])
    # set of PointGroupElements
    symset = {e}
    # set of generators
    genset = set()
    symmetry_candidates -= symset
    not_symmetries = set()
    n = 0
    while symmetry_candidates:
        # For reproducibility, iterate over elements in sorted order
        # instead of simply popping an arbitrary element
        g = min(symmetry_candidates)
        symmetry_candidates -= {g}
        # Find unitary part
        gr = _find_unitary(model, Ps, g, sparse=sparse_linalg)
        if gr.U is not None:
            # Check that it's indeed a symmetry
            assert gr.apply(model) == model, (n, gr)
            genset.add(gr)
            # Needless to test anything in the group generated by the
            # symmetries found already, they are symmetries for sure.
            symset = generate_group({gr} | genset)
            symmetry_candidates -= symset
            # Needless to test anything of the form Q*R, Q*R**-1,
            # R*Q, R**-1*Q where Q is a symmetry and R is not,
            # it is surely not a symmetry.
            # higher powers of R may still be symmetries.
            new_ns = set_multiply({g, g.inv()}, not_symmetries)
            new_ns -= not_symmetries
            new_ns |= set_multiply(new_ns, {g, g.inv()})
            not_symmetries |= new_ns
        else:
            new_ns = set_multiply({g, g.inv()}, symset)
            new_ns -= not_symmetries
            new_ns |= set_multiply(symset, new_ns)
            not_symmetries |= new_ns
        symmetry_candidates -= not_symmetries
        n+=1
    if verbose:
        print('{} symmetries explicitely tested of {} candidates.'.format(n, m))
    assert not any(g.U is None for g in symset)
    if generators:
        return genset, Ps
    else:
        return symset, Ps


def _find_unitary(model, Ps, g, sparse=False, checks=False):
    """Test if the candidate k-space symmetry is (anti)unitary (anti)symmetry,
    if not, return 'None', if yes, return the unitary part of the transformation
    'U'. Checked condition if unitary: U H(inv(R) k) = (+/-) H(k) U
    Checked condition if antiunitary: U H(-inv(R) k).conj() = (+/-) H(k) U.
    RHS (+/-) stands for symmetry/antisymmetry.

    Parameters:
    -----------
    model : Model
        model which represents family of Hamiltonians
    Ps : iterable of ndarrays
        Projectors onto the irreducible subspaces of on-site symmetries,
        as returned by '_reduce_hamiltonian'
    g : PointGroupElement
        Standard representation of symmetry operator. g.U must be None.
    checks : bool
        Whether to perform checks.

    Returns:
    --------
    gr : PointGroupElement
        Point group operator with gr.U set to the Hilbert space action
        of the symmetry is found, otherwise identical to g.
    """
    if g.U is not None:
        raise ValueError('g.U must be None.')
    Rmodel = g.apply(model)
    if set(model) != set(Rmodel):
        return g
    HR, HL = [], []
    # Only test eigenvalues if all matrices are Hermitian
    ev_test = True
    for key, matL in model.items():
        HR.append(matL)
        matR = Rmodel[key]
        HL.append(matR)
        ev_test = ev_test and allclose(matL, matL.T.conj()) and allclose(matR, matR.T.conj())
    HR, HL = np.array(HR), np.array(HL)
    # Need to carry conjugation on left side through P
    if g.conjugate:
        PsL = [P.conj() for P in Ps]
    else:
        PsL = Ps
    HRs = [mtm(P[0].T.conj(), HR, P[0]) for P in Ps]
    HLs = [mtm(P[0].T.conj(), HL, P[0]) for P in PsL]

    squares_to_1 = g * g == g.identity()
    block_dict = _find_unitary_blocks(HLs, HRs, Ps, conjugate=g.conjugate, ev_test=ev_test,
                                      squares_to_1=squares_to_1, sparse=sparse)
    S = _construct_unitary(block_dict, Ps, conjugate=g.conjugate, squares_to_1=squares_to_1)

    if checks:
        for i, j in it.product(range(len(Ps)), range(len(Ps))):
            for a, b in it.product(range(len(Ps[i])), range(len(Ps[j]))):
                if i !=j or a!=b:
                    assert np.allclose(mtm(PsL[i][a].T.conj(), HL, PsL[j][b]), 0)
                    assert np.allclose(mtm(Ps[i][a].T.conj(), HR, Ps[j][b]), 0)
                else:
                    assert allclose(mtm(PsL[i][a].T.conj(), HL, PsL[j][b]), HLs[i])
                    assert allclose(mtm(Ps[i][a].T.conj(), HR, Ps[j][b]), HRs[i])
        if (not g.conjugate) and (not g.antisymmetry) and (S is not None):
            assert allclose(S @ HL, HR @ S)

    return PointGroupElement(g.R, g.conjugate, g.antisymmetry, S)


def _find_unitary_blocks(HLs, HRs, projectors, squares_to_1=True, conjugate=False,
                         ev_test=True, sparse=False):
    """Find candidate symmetry linear spaces in all blocks.
    HLs and HRs are lists of reduced Hamiltonians (families) that go to left and right side
    of the equations.

    Returns a dictionary {(i, j): Uij} of all symmetry candidate blocks that have a
    nonzero solution of Uij @ HLs[j] = HRs[i] @ Uij for Uij.

    If squares_to_1=True, it is assumed that the operators square is proportional to 1
    in every block. The search is limited to j <= i, the diagonal blocks have a phase choice
    and the off-diagonal blocks with j > i are constructed to ensure squaring to +-1.
    Otherwise the blocks Uij and Uji are calculated independently.

    If ev_test=True the eigenvalues of the matrices are tested first
    """
    # Only need to find symmetries in half of each block of the Hamiltonian.
    # We take blocks in the lower triangular half and on the diagonal.
    block_dict = {}
    ind = range(len(projectors))
    # Pretest eigenvalues
    if ev_test:
        evsL = [[la.eigvalsh(h) for h in HLs[i]] for i in ind]
        evsR = [[la.eigvalsh(h) for h in HRs[i]] for i in ind]
    for (i, j) in it.product(ind, ind):
        # Only do j <= i if squares to 1
        if squares_to_1 and j>i:
            continue
        # Only allowed between blocks of identical shape
        if projectors[i].shape != projectors[j].shape:
            continue
        # Pretest eigenvalues
        if ev_test:
            if not allclose(evsL[j], evsR[i]):
                continue
        # Find block ij of the symmetry operator
        block_dsymm = solve_mat_eqn(HLs[j], HRs[i], hermitian=False, traceless=False, sparse=sparse, k_max=2)
        # Normalize block_dsymm such that it is close to unitary. The matrix
        # returned by solve_mat_eqn is normalized such that Tr(X.T.conj() @ X) is close to 1.
        block_dsymm = np.sqrt(block_dsymm.shape[-1]) * block_dsymm
        # If the space is not empty, we store it and the indices of the block.
        if len(block_dsymm):
            # There should be only one solution, which is invertible
            if len(block_dsymm) > 1 or np.isclose(la.det(block_dsymm[0]), 0):
                raise ValueError('Hamiltonian blocks have residual symmetry.')
            block_dsymm = block_dsymm[0]
            assert allclose(block_dsymm @ HLs[j],
                               HRs[i] @ block_dsymm)
            # The block must be proportional to a unitary
            prop_to_I, coeff = prop_to_id(block_dsymm.dot(block_dsymm.T.conj()))
            assert prop_to_I and np.isclose(np.imag(coeff), 0) and np.real(coeff)>0
            # Normalize such that it is unitary
            block_dsymm = block_dsymm/np.sqrt(coeff)
            block_dict[(i, j)] = block_dsymm
            # If squares to 1, fill out the lower triangle
            if squares_to_1:
                block_dict[(i, j)], block_dict[(j, i)] = _nice_square(block_dsymm, (i == j), conjugate)
    return block_dict


def _nice_square(block_dsymm, diagonal, conjugate):
    # Make sure blocks square to +-1
    # Diagonal blocks need proper phase choice if unitary,
    # nothing to be done if antiunitary, must square to +-1
    if diagonal:
        if conjugate:
            prop_to_I, coeff = prop_to_id(block_dsymm.dot(block_dsymm.conj()))
            assert prop_to_I and (np.isclose(coeff, 1) or np.isclose(coeff, -1))
        else:
            prop_to_I, coeff = prop_to_id(block_dsymm.dot(block_dsymm))
            assert prop_to_I and np.isclose(np.abs(coeff), 1)
            block_dsymm = block_dsymm/np.sqrt(coeff)
        return block_dsymm, block_dsymm
    # Off-diagonal blocks are chosen such that it squares to +1
    else:
        if conjugate:
            return block_dsymm, block_dsymm.T
        else:
            return block_dsymm, block_dsymm.T.conj()


def _construct_unitary(block_dict, projectors, conjugate=False, squares_to_1=True):
    """Search for combinations of blocks of the symmetry operator that when combined give a symmetry
    in canonical form, i.e. with only one nonzero block per row and column, and attempt to construct
    the operator. """
    block_keys = block_dict.keys()
    n = len(projectors)
    # Need to find a canonical form of the symmetry operator.
    # Iterate over all combinations of the nonzero blocks
    for perm in it.combinations(block_keys, len(projectors)):
        # Make the corresponding matrix
        M = np.zeros((n, n))
        for i, j in perm:
            M[i, j] = 1
        # Check that it is a permutation matrix
        if not np.all(M @ M.T == np.eye(n)):
            continue
        # If squares_to_1 check that (j, i) block is also present
        if squares_to_1 and not np.all(M == M.T):
            continue
        # Construct the symmetry operator and return
        # Initialize complete symmetry operator
        S = np.zeros((projectors[0].shape[-2], projectors[0].shape[-2]), dtype=complex)
        # Iterate over all blocks that give a canonical form
        for i, j in perm:
            block = block_dict[(i, j)]
            # Rebuild full operator using projectors
            pi, pj = projectors[i], projectors[j]
            # Use conjugate projector if antiunitary
            if conjugate:
                # 'aij, jk, alk -> il'
                S += np.tensordot(pi @ block, pj, axes=((0, 2), (0, 2)))
            else:
                S += np.tensordot(pi @ block, pj.conj(), axes=((0, 2), (0, 2)))
        assert prop_to_id(S.dot(S.T.conj()))[0]
        return S
    # If we cannot construct a canonical symmetry operator, return None
    return None

### Continuous spatial symmetry finding

def continuous_symmetries(model, Ps=None, prettify=True, num_digits=8, sparse_linalg=False):
    """Find continuous rotation symmetries of Hamiltonian family represented
    by model. Hamiltonian is reduced, so on-site continuous symmetries
    are factored out.

    Parameters
    ----------
    model : Model
        symbolic representation of the Hamiltonian family
    Ps : ndarray, optional
        Projectors as returned by _reduce_hamiltonian
    prettify : bool
        Whether to carry out sparsification of the results, in general an
        arbitrary linear combination of the symmetry generators is returned.
    num_digits : float
        Absolute precision when deciding whether symmetry leaves Hamiltonian
        invariant and prettifying the result.
    sparse_linalg : bool
        Whether to use sparse linear algebra in the calculation.
        Can give large performance gain in large systems.

    Returns
    -------
    symmetries : list of ContinuousGroupGenerator
        List of linearly independent symmetry generators.
    """
    if not Ps:
        Ps = _reduce_hamiltonian(np.array(list(model.values())), sparse=sparse_linalg)
    reduced_hamiltonians = _reduced_model(model, Ps)
    dim = len(model.momenta)
    if dim <= 1:
        # There is no continuous rotation in 0 and 1 D
        return []
    Rs = lambda: matrix_basis(dim, antihermitian=True, real=True)
    # length of Rs
    NR = dim * (dim - 1) / 2
    blockdims = [list(rham.values())[0].shape[0] for rham in reduced_hamiltonians]
    # Blocks corresponding to real space rotations
    Rblocks = []
    # Blocks corresponding to Hilbert space transformations
    Lblocks = []
    # Iterate over all the reduced hamiltonian blocks
    for rham in reduced_hamiltonians:
        blockdim = rham.shape[0]
        L = None
        # Generate all reduced hamiltonians transformed by spatial part
        trf_hams = [ContinuousGroupGenerator(1j * R, L).apply(rham) for R in Rs()]
        # Generate all keys that appear as transformed reduced hamiltonians
        keys = rham.keys()
        for th in trf_hams:
            keys |= th.keys()
        keys = list(keys)

        # Iterate over all reduced hamiltonians transformed by only spatial part
        Rblock = family_to_vectors(trf_hams, all_keys=keys).T
        Rblocks.append(Rblock)

        # Iterate over all reduced hamiltonians transformed by only hilbert space part
        R = None
        Ls = matrix_basis(blockdim, traceless=True)
        trf_hams = [ContinuousGroupGenerator(R, L).apply(rham) for L in Ls]
        Lblock = family_to_vectors(trf_hams, all_keys=keys).T
        Lblocks.append(Lblock)

    # Build constraint matrix: first the spatial blocks in a column, then the
    # Hilbert-space blocks as block-diagonal, as they do not mix different
    # reduced subspaces.
    constraints = np.hstack(((np.vstack(Rblocks), la.block_diag(*Lblocks))))

    # Find the linearly independent solutions
    null_vecs = nullspace(constraints, sparse=sparse_linalg)
    if prettify:
        null_vecs = sparse_basis(null_vecs, reals=False, num_digits=num_digits)

    # Build the symmetry generator
    symmetries = []
    for v in null_vecs:
        # Build spatial part
        R = np.sum((1j * R * v[i] for i, R in enumerate(Rs())), axis=0)
        # Build unitary part for each block
        L = np.zeros((Ps[0].shape[1], Ps[0].shape[1]), dtype=complex)
        for i, rham in enumerate(reduced_hamiltonians):
            # There is no action in a 1D block
            if blockdims[i] > 1:
                Ls = matrix_basis(blockdims[i], traceless=True)
                blockind = int(NR + np.sum([d**2-1 for d in blockdims[:i]]))
                l = np.sum((l * v[blockind + j] for j, l in enumerate(Ls)), axis=0)
                L += np.einsum('aij,jl,akl->ik', Ps[i], l, Ps[i].conj())
        g = ContinuousGroupGenerator(R, L)
        symmetries.append(g)
        # Check that it is a symmetry
        trf = g.apply(model)
        trf.restructure(atol=10**(-num_digits+1))
        assert trf == {}, (trf, g)
    return symmetries


def _reduced_model(model, Ps=None):
    """
    Construct reduced Hamiltonians in monomial form.

    Parameters
    ----------
    model : Model
        symbolic representation of the Hamiltonian family
    Ps : list fo 3 index ndarrays
        projectors 'Ps' returned '_reduce_hamiltonian()'
        Optional, can be provided to speed up the calculation.

    Returns
    -------
    reduced_hamiltonians : list of Model
        List of reduced Hamiltonian families, each projected
        on the symmetry irreducible subspaces.
    """
    if Ps is None:
        Ps = _reduce_hamiltonian(np.array([H for H in model.values()]))
    reduced_hamiltonians = []
    for P in Ps:
        Hr = P[0].T.conj() * model * P[0]
        reduced_hamiltonians.append(Hr)
    return reduced_hamiltonians
