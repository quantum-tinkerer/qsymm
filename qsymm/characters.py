# +
import numpy as np
import sympy
import qsymm
from itertools import product
from qsymm.linalg import split_list, allclose, commutator, simult_diag, mtm, solve_mat_eqn
from qsymm.symmetry_finder import symmetry_adapted_sun
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import scipy.sparse as scsp
import scipy
from copy import copy
from functools import cached_property

sympy.init_printing(print_builtin=True)
np.set_printoptions(precision=2, suppress=True, linewidth=150)

# %load_ext autoreload
# %autoreload 2

# +
def conjugate_classes(group):
    # make sure the identity is the first class
    e = next(iter(group)).identity()
    conjugate_classes = [{e}]
    class_by_element = {e: 0}
    rest = set(group) - {e}
    i = 1
    while rest:
        # use sorting for reproducibility
        g = min(rest)
        conjugates = {h * g * h.inv() for h in group}
        conjugate_classes.append(conjugates)
        rest -= conjugates
        class_by_element |= {h: i for h in conjugates}
        i += 1
    conjugate_classes = np.array(conjugate_classes)
    sort_order = np.argsort(list(map(len, conjugate_classes)))
    conjugate_classes = conjugate_classes[sort_order]
    class_representatives = [min(cl) for cl in conjugate_classes]
    class_by_element = {g: np.argsort(sort_order)[c] for g, c in class_by_element.items()}
    return conjugate_classes, class_representatives, class_by_element

def build_M_matrices(group, conjugate_classes, class_by_elemet):
    k = len(conjugate_classes)
    M = np.zeros((k ,k ,k), dtype=int) # r, s, t
    class_reps = [min(c) for c in conjugate_classes]
    for x, y in product(group, repeat=2):
        z = x * y
        if z in class_reps:
            M[class_by_elemet[x], class_by_elemet[z], class_by_elemet[y]] +=1
    # transform to a basis where these are normal matrices
    A = np.diag(np.array([len(c)**(1/2) for c in conjugate_classes]))
    Ai = np.diag(np.array([len(c)**(-1/2) for c in conjugate_classes]))
    M = mtm(A, M, Ai)
    assert allclose([commutator(m, m.conj().T) for m in M], 0)
    # They are mutually commuting
    assert allclose([commutator(m1, m2) for m1, m2 in product(M, repeat=2)], 0)
    return M

# def grouped_diag(H, tol=1e-6):
#     # Group the eigenvalues and eigenvectors of matrix H
#     # such that approximately equal eigenvalues are grouped together.
#     # Returns the grouped eigenvalues, eigenvectors
#     evals, U = np.linalg.eig(H)
#     # Treat complex eigenvalues as 2D vectors
#     evvec = np.array([evals.real, evals.imag]).T
#     # Find connected clusters of close values
#     con = cdist(evvec, evvec) < tol/len(H)
#     _, groups = connected_components(con)
#     U = [np.linalg.qr(U[:, groups == i])[0] for i in range(max(groups+1))]
#     evals = np.array([evals[groups == i][0] for i in range(max(groups+1))])
#     return evals, U

def subspace_intersection(u1, u2, tol=1e-6):
    """Calculate a basis for the intersection of subspaceces
    given by the orthonormal sets of column vectors u1 and u2."""
    assert allclose(u1.conj().T @ u1, np.eye(u1.shape[1]))
    assert allclose(u2.conj().T @ u2, np.eye(u2.shape[1]))
    A = u2.conj().T @ u1
    U, S, Vh = np.linalg.svd(A)
    ind = np.argwhere(np.isclose(S, 1, atol=tol)).flatten()
    if len(ind) == 0:
        return None
    # test that they span the same subspace
    A_reduced = (u2 @ U[:, ind]).T.conj() @ (u1 @ Vh.T.conj()[:, ind])
    assert allclose(A_reduced @ A_reduced.T.conj(), np.eye(A_reduced.shape[0]), atol=tol), (U, S, Vh)
    return u1 @ Vh.T.conj()[:, ind]

def common_eigenvectors(mats, tol=1e-6, quit_when_1d=False):
    eigensubspaces = [np.eye(mats[0].shape[0])]
    for i in range(len(mats)):
        _, new_subspaces = grouped_diag(mats[i], tol=tol)
        new_eigensubspaces = [subspace_intersection(u1, u2, tol=tol)
                          for u1, u2 in product(eigensubspaces, new_subspaces)]
        new_eigensubspaces = [s for s in new_eigensubspaces if s is not None]
        if not len(new_eigensubspaces) == len(eigensubspaces):
            # don't change anything if we didn't split anything
            eigensubspaces = new_eigensubspaces
            print(len(eigensubspaces))
            print([s.shape for s in eigensubspaces])
        if all(s.shape[1] == 1 for s in eigensubspaces) and quit_when_1d:
            break
    return eigensubspaces

def character_table(group, conjugate_cl=None, class_by_element=None, tol=1e-9):
    # Using Burnside's method, based on DIXON Numerische Mathematik t0, 446--450 (1967)
    if conjugate_cl is None or class_by_element is None:
        conjugate_cl, _, class_by_element = conjugate_classes(group)
    class_sizes = np.array([len(c) for c in conjugate_cl])
    Ai = np.diag(class_sizes**(-1/2))
    M = build_M_matrices(group, conjugate_cl, class_by_element)
    chars = np.hstack(simult_diag(M, tol))
    chars = Ai @ chars
    chars = chars.T
    norms = character_product(chars, chars, class_sizes, check_int=False)
    chars = np.sqrt(1 / norms[:, None]) * chars
    # Make sure all characters of the identity is positive real
    chars *= (chars[:, 0].conj() / np.abs(chars[:, 0]))[:, None]
    # Sort the characters for reproducible result
    chars = sort_characters(chars)
    assert allclose(chars @ np.diag(class_sizes) @ chars.T.conj() / sum(class_sizes), np.eye(chars.shape[0]))
    assert chars.shape[0] == chars.shape[1], chars.shape
    return chars

def sort_characters(characters):
    # Sort the characters for reproducible result with trivial rep first
    # and a small imaginary shift so complex reps are also sorted reproducibly
    sort_order = np.lexsort(np.round(np.abs(characters.T[::-1] - 1 - 0.1j), 3))
    return characters[sort_order, :]

def character_product(char1, char2, class_sizes, check_int=True):
    prod = np.sum(class_sizes[..., :] * char1 * char2.conj(), axis=-1) / sum(class_sizes)
    if not check_int:
        return prod
    prod_round = np.around(prod).real.astype(int)
    if not allclose(prod, prod_round):
        raise ValueError('Invalid characters, the product should be integer.')
    return prod_round

def decompose_representation(group, use_R=False, irreps=None,
                             conjugate_cl=None, class_reps=None, class_by_element=None):
    if any(a is None for a in [conjugate_cl, class_reps, class_by_element]):
        conjugate_cl, class_reps, class_by_element = conjugate_classes(group)
    class_sizes = np.array([len(c) for c in conjugate_cl])
    char = np.array([np.trace(g.R) if use_R else np.trace(g.U) for g in class_reps])
    if irreps is None:
        irreps = character_table(group, tol=1e-6, conjugate_cl=conjugate_cl, class_by_element=class_by_element)
    return character_product(char, irreps, class_sizes)

def order(g):
    n = 1
    h = g
    identity = g.identity()
    if g.RSU2 is not None:
        full_rot = g.identity()
        full_rot.RSU2 = -np.eye(2)
    while True:
        if h == identity or (g.RSU2 is not None and h == full_rot):
            return n
        n += 1
        h *= g

def find_generators(group):
    """Try to find a small generator set of group. Not guaranteed
    to find the minimal set, uses greedy algorithm by including the
    highest order elements first."""
    group_list = list(group)
    group_list.sort()
    group_list.sort(key=order)
    group_list.reverse()
    generators = set()
    current_group = set()
    for g in group_list:
        new_group = generate_group(generators | {g})
        if len(new_group) > len(current_group):
            current_group = new_group
            generators |= {g}
        if len(new_group) == len(group):
            return generators
    else:
        raise ValueError("Generator finding failed, `group` does not appear to be a closed group.")

def check_U_consistency(group):
    """Check that the unitary parts have consistent phases."""
    # Way to retrieve the representative U
    group_dict = {g: g for g in group}
    # Brute force check of full multiplication table
    for g, h in product(group, repeat=2):
        if not allclose(group_dict[g * h].U, (g * h).U):
            return False
    else:
        return True


# from functools import property
from qsymm.groups import generate_group, PointGroupElement, _mul, _eq
from qsymm.linalg import prop_to_id
import tinyarray as ta
from copy import copy

class PointGroup(set):

    def __init__(self, generators, double_group=None, _tests=False, tol=1e-9):
        """Class to store point group objects and related representation
        theoretical information. Only supports PointGroupElements.
        It represents the discrete group generated by a set of generators,
        these can be accessed through a set interface."""

        super().__init__(generators)
        self.generators = generators

        if not all(isinstance(g, PointGroupElement) for g in generators):
            raise ValueError('Only iterables of PointGroupElements is supported.')

        antiunitary_generators = [g for g in generators if g.conjugate]
        if len(antiunitary_generators) == 0:
            self.antiunitary_generator = None
        # Make the antiunitary generator same as the one in generators,
        # except if it doesn't square to identity or there are several
        elif (len(antiunitary_generators) == 1 and
              abs(prop_to_id((antiunitary_generators[0]**2).R)[1] - 1) < 1e-5):
            self.antiunitary_generator = antiunitary_generators[0]
        else:
            antiunitaries = [g for g in self.elements if g.conjugate]
            antiunitaries_square_to_1 = [g for g in antiunitaries if abs(prop_to_id((g**2).R)[1] - 1) < 1e-5]
            if len(antiunitaries_square_to_1) == 0:
                raise NotImplementedError('Only antiunitaries that square to identity rotation are supported.')
            self.antiunitary_generator = min(antiunitaries_square_to_1)

        all_dg = all(g.RSU2 is not None for g in generators)
        any_dg = any(g.RSU2 is not None for g in generators)
        if all_dg != any_dg:
            raise ValueError('The `RSU2` attribute must be set for either all generators or none.')
        if double_group is None:
            self.double_group = all_dg
        elif double_group and not all_dg:
            raise ValueError('To use `double_group=True`, all generators must have the `RSU2` attribute set.')
        else:
            self.double_group = double_group

        self.U_set = all(g.U is not None for g in self.elements)
        if self.U_set:
            self.U_shape = next(iter(generators)).U.shape

        self._tests = _tests
        self.tol = tol

    @cached_property
    def elements(self):
        return generate_group(self)

    @cached_property
    def elements_list(self):
        return sorted(list(self.elements))

    @cached_property
    def unitary_elements(self):
        return [g for g in generate_group(self) if not g.conjugate]

    @cached_property
    def unitary_elements_list(self):
        return sorted(list(self.unitary_elements))

    @cached_property
    def minimal_generators(self):
        """Find minimal set of unitary generators."""
        minimal_generators = find_generators(self.unitary_elements)
        unitary_generators = {g for g in self.generators if not g.conjugate}
        if len(minimal_generators) >= len(unitary_generators):
            return unitary_generators
        else:
            return minimal_generators

    @cached_property
    def consistent_U(self):
        if not self.U_set:
            raise ValueError('The U attribute must be set for all goup elements.')
        return check_U_consistency(self.elements)

    def fix_U_phases(self):
        """
        Fix phases of unitaries such that the (double) point group generated by generators
        forms a true representation. This changes the PointGroupElements in the PointGroup in-place.
        """
        if self.consistent_U:
            return
        gens_orders = []
        for g in self.minimal_generators:
            n = order(g)
            gn = g**n
            ### TODO: generalize this to work with little group representations
            sign = 1 if gn.RSU2 is None else prop_to_id(gn.RSU2)[1]
            ppi, phase = prop_to_id(gn.U)
            phase = np.angle(phase)
            if not ppi:
                raise NotImplementedError('The PointGroup appears to contain nontrivial conserved quantities, '+
                                          'this case is not supported.')
            gens_orders.append((g, n, phase, sign))

        fixes = []

        for ms in product(*[range(n) for _, n, _, _ in gens_orders]):
            # print('-', end='')
            new_gens = []
            for (g, n, phase, sign), m in zip(gens_orders, ms):
                g_new = copy(g)
                g_new.U = g_new.U * np.exp(1j * (-phase/n + 2 * np.pi / n * (m - (sign-1)/4)))
                new_gens.append(g_new)
            if check_U_consistency(generate_group(new_gens)):
                self.minimal_generators = new_gens
                self.unitary_elements = generate_group(self.minimal_generators)
                self.generators = {g for g in self.unitary_elements if g in self}
                ### TODO: is there a more elegant way to update set contents?
                super().__init__(self.generators)
                self.consistent_U = True
                break
        else:
            raise ValueError('Phase fixing failed! Try changing the `double_group` setting.')

    def _set_conjugate_classes(self):
        (self.conjugate_classes,
         self.class_representatives,
         self.class_by_element) = conjugate_classes(self.unitary_elements)

    @cached_property
    def conjugate_classes(self):
        self._set_conjugate_classes()
        return self.conjugate_classes

    @cached_property
    def class_by_element(self):
        self._set_conjugate_classes()
        return self.class_by_element

    @cached_property
    def class_representatives(self):
        self._set_conjugate_classes()
        return self.class_representatives

    @cached_property
    def character_table(self):
        r"""
        Return the character table of the unitary part of the group.
        Rows correspond to the different irreps, and columns to the conjugacy
        classes in the order of `self.conjugacy_classes`.
        """
        return character_table(self.unitary_elements, self.conjugate_classes, self.class_by_element, tol=self.tol)

    @cached_property
    def character_table_full(self):
        r"""
        Return the character table of the unitary part of the group.
        Rows correspond to the different irreps, and the character is listed
        for all elements in the order of `self.unitary_elements_list`.
        """
        return self.character_table[:, np.array([self.class_by_element[g] for g in self.unitary_elements_list])]

    @cached_property
    def decompose_U_rep(self):
        if not self.consistent_U:
            self.fix_U_phases()
        return decompose_representation(self.unitary_elements, use_R=False,
                                        irreps=self.character_table,
                                        conjugate_cl=self.conjugate_classes,
                                        class_reps=self.class_representatives,
                                        class_by_element=self.class_by_element)

    @cached_property
    def decompose_R_rep(self):
        return decompose_representation(self.unitary_elements, use_R=True,
                                        irreps=self.character_table,
                                        conjugate_cl=self.conjugate_classes,
                                        class_reps=self.class_representatives,
                                        class_by_element=self.class_by_element)

    @cached_property
    def symmetry_adapted_basis(self):
        """Find a symmetry adapted basis of the unitary representation in U.
        Returns a list of sets of basis vectors, each set spanning an
        invariant subspace. The ordering corresponds to the order
        nonzero weight irreps appear in `decompose_U_rep`. The division
        of subspaces belonging to the same irrep is not unique."""
        ### TODO: Add support for antiunitary symmetries.
        bases = []
        for chi, n in zip(self.character_table_full, self.decompose_U_rep):
            if n == 0:
                continue
            d = int(np.around(chi[0]).real)
            basis_chi = np.empty((self.U_shape[0], 0))
            for v in np.eye(self.U_shape[0]):
                w = np.sum([chi[i].conj() * g.U @ v for i, g in enumerate(self.unitary_elements_list)], axis=0)
                w *= chi[0] / len(self.unitary_elements)
                if np.linalg.norm(w) <= self.tol:
                    continue
                if n==1 and d==1:
                    for i, g in enumerate(self.unitary_elements_list):
                        assert allclose(chi[i] * w, g.U @ w)
                wspan = np.array([g.U @ w for g in self.unitary_elements]).T
                basis_chi = np.hstack([basis_chi, wspan])
                rank = np.linalg.matrix_rank(basis_chi, self.tol)
                assert rank <= n * d, (rank, n, d)
                if rank == n * d:
                    break
            basis_chi = scipy.linalg.qr(basis_chi, pivoting=True)[0]
            basis_chi = basis_chi[:, :rank]

            # This results in a nice basis where first generator is diagonal,
            # the others are real in the off-diagonal blocks as much as possible
            gens = np.array([basis_chi.T.conj() @ g.U @ basis_chi for g in self.minimal_generators])
            vecs = symmetry_adapted_sun(gens, n=n)
            for i in range(n):
                bases.append(basis_chi @ vecs[:, :, i].T)
        return bases

    @cached_property
    def regular_representation(self):
        """Construct the regular representation of the unitary part
        of the group with permutation matrices for U."""
        ### TODO: allow sparse matrices for U in PGE to speed this up
        new_generators = set()
        element_dict = {g: i for i, g in enumerate(self.unitary_elements)}
        for g in self.minimal_generators:
            mat = np.zeros((len(self.unitary_elements), len(self.unitary_elements)), dtype=int)
            for h in self.unitary_elements:
                mat[element_dict[h], element_dict[g*h]] = 1
            new_g = copy(g)
            new_g.U = mat
            new_generators.add(new_g)
        reg_rep = type(self)(new_generators)
        if self._tests:
            assert reg_rep.consistent_U
            assert allclose([g.R for g in self.class_representatives], [g.R for g in reg_rep.class_representatives])
            assert allclose(reg_rep.character_table, self.character_table)
        reg_rep.character_table = self.character_table
        assert allclose(reg_rep.decompose_U_rep, reg_rep.character_table[:, 0])
        return reg_rep

    def antiunitary_square(self):
        # return the square of the antiunitary generator including all phases
        # only returns a complex phase, we assume TR^2 is a pure phase
        TR = self.antiunitary_generator
        if TR is None:
            raise ValueError('Group contains no antiunitary.')
        TR2 = 1 if TR.conjugate is True else prop_to_id((TR**2).RSU2)[1]
        assert abs(TR2**2 - 1) < 1e-6
        TR2 = int(np.around(TR2).real)
        return TR2

    @cached_property
    def irreps(self):
        """Construct a matrix representation for every irrep
        of the unitary part of the group."""
        reg_rep = self.regular_representation
        irreps = []
        bases = reg_rep.symmetry_adapted_basis
        m = 0
        # print(reg_rep.decompose_U_rep)
        for i, n in enumerate(reg_rep.decompose_U_rep):
            basis_chi = bases[m]
            new_generators = set()
            for g in reg_rep.minimal_generators:
                new_g = copy(g)
                new_g.U = basis_chi.T.conj() @ g.U @ basis_chi
                assert allclose(np.trace(new_g.U), reg_rep.character_table_full[i, reg_rep.unitary_elements_list.index(g)])
                if self._tests:
                    assert allclose(g.U.T.conj() @ g.U, np.eye(g.U.shape[1]))
                    assert allclose(new_g.U.T.conj() @ new_g.U, np.eye(new_g.U.shape[1]))
                new_generators.add(new_g)
            irrep = type(self)(new_generators)
            assert len(self.unitary_elements) == len(irrep.unitary_elements)
            assert irrep.class_representatives == reg_rep.class_representatives
            if self._tests:
                assert irrep.consistent_U
                assert allclose(irrep.character_table, reg_rep.character_table)
                assert allclose(irrep.decompose_U_rep, np.eye(reg_rep.character_table.shape[0])[i])
            irrep.character_table = reg_rep.character_table
            irrep.character_table_full = reg_rep.character_table_full
            irrep.decompose_U_rep = np.eye(reg_rep.character_table.shape[0])[i]
            irreps.append(irrep)
            m += n
        if self.antiunitary_generator is None:
            # we are done if everything is unitary
            return irreps

        # Find what TR squares to
        TR = self.antiunitary_generator
        TR2 = self.antiunitary_square()

        # Find conjugate pairs of irreps
        chars = self.character_table_full
        # Make product with conjugate
        conj_prod = chars @ chars.T / chars.shape[1]
        conj_ind = zip(*np.nonzero(np.triu(np.around(conj_prod))))

        physical_irreps = []
        # construct the irreps with TR
        for i, j in conj_ind:
            if i == j and irreps[i].reality == TR2:
                # real or pseudoreal irrep, no need to double
                new_generators = irreps[i].minimal_generators
                # just need to find the TR operator
                # TRU @ U(g)^* = U(g) @ TRU
                right = np.array([g.U for g in irreps[i].minimal_generators])
                TRU = solve_mat_eqn(right.conj(), right)
                assert TRU.shape[0] == 1
                TRU = TRU[0]
                TRU = TRU / np.sqrt(prop_to_id(TRU @ TRU.conj())[1])
                assert abs(prop_to_id(TRU @ TRU.conj())[1] - TR2) < 1e-6
                new_TR = copy(TR)
                new_TR.U = TRU
                new_generators.add(new_TR)

            else:
                # If TR^2 = -1, but full rotation is represented as +1,
                # it is not possible to construct irrep
                if TR2 == -1:
                    full_rotation = next(iter(irreps[i].minimal_generators)).identity()
                    full_rotation.RSU2 = -full_rotation.RSU2
                    full_rotation = [g for g in irreps[i].elements if g == full_rotation]
                    assert len(full_rotation) == 1
                    full_rotation = full_rotation[0]
                    if not allclose(full_rotation.U, -np.eye(full_rotation.U.shape[0])):
                        continue

                # Need to double it and TR maps between copies
                new_generators = set()
                for g in irreps[i].minimal_generators:
                    new_g = copy(g)
                    new_g.U = scipy.linalg.block_diag(g.U, g.U.conj())
                    new_generators.add(new_g)
                # Make TR to correct square
                new_TR = copy(TR)
                new_TR.U = np.kron([[0, 1], [TR2, 0]], np.eye(irreps[i].U_shape[0]))
                new_generators.add(new_TR)

            irrep = type(self)(new_generators)
            if self._tests:
                assert irrep.consistent_U
            irrep.character_table = self.character_table
            physical_irreps.append(irrep)
        return physical_irreps

    @cached_property
    def reality(self):
        """Determine the reality of the unitary representation:
        1 for real, 0 for complex, -1 for pseudoreal.
        Only works for irreducible representations."""
        rep = self.decompose_U_rep
        if not sum(rep) == 1:
            raise ValueError('Reality is only defined for irreducible representations.')
        rep = rep @ self.character_table
        # This is the same as the formula below, but also works for projective representations
        # without refering to the factor system
        # np.sum([g.factor(g) * rep[self.class_by_element[g**2]] for g in self.unitary_elements])
        reality = np.sum([np.trace(g.U @ g.U) for g in self.unitary_elements])
        reality = reality/len(self.unitary_elements)
        # This assumes that they are related by an antiunitary that squares to ±1, is this always true?
        assert reality - np.around(reality) < 1e-6
        return np.around(reality).real.astype(int)


class SpaceGroupElement(PointGroupElement):
    def __init__(self, R, t, periods, conjugate=False, antisymmetry=False, U=None, RSU2=None,
                 _strict_eq=False, *, locals=None):
        """Container for space group elements. The primitive translation vectors of the
        enclosing space group are `periods`, the translation part of this element is `t`.
        Equality is only checked up to translations."""
        # Allow initialization with a PGE as R, then all other optional parameters are ignored
        if isinstance(R, PointGroupElement):
            if conjugate or antisymmetry or U is not None or RSU2 is not None or _strict_eq:
                raise ValueError('When initializing with a PointGroupElement, no optional arguments can be provided.')
            super().__init__(R.R, R.conjugate, R.antisymmetry, R.U, R.RSU2, R._strict_eq, locals=locals)
        else:
            super().__init__(R, conjugate, antisymmetry, U, RSU2, _strict_eq, locals=locals)
        # Check that R is compatible with periods
        # Transform R to lattice vector basis
        self.periods = np.atleast_2d(periods)
        self._R_trf = np.dot(np.linalg.inv(periods).T, np.dot(self.R, periods.T))
        if not allclose(np.around(self._R_trf), self._R_trf):
            raise ValueError('Rotation is incompatible with lattice periods.')
        self._R_trf = ta.array(np.around(self._R_trf), int)
        self.t = ta.array(t)

    # Implement multiplication
    def __mul__(self, g2):
        # This also works for antiunitaries as long as the antiunitary part commutes
        # with the spatial part, i.e. the antiunitary is spatially local.
        g1 = self
        if not allclose(g1.periods, g2.periods):
            raise ValueError('Multiplication is only allowed for SpaceGroupElements with the same `periods`.')
        R1, t1 = g1.R, g1.t
        R2, t2 = g2.R, g2.t
        # Translation part of product
        t = t1 + _mul(R1, t2)
        # Delegate most of the work to PGE.__mul__
        return SpaceGroupElement(PointGroupElement.__mul__(g1, g2), t, g1.periods)

    # Implement equality testing ignoring integer translations
    # Same as PGE equality, but need extra check that t's only differ by integer,
    # otherwise raise error because periods are not primitive.
    def __eq__(self, other):
        if not _eq(ta.array(self.periods), ta.array(self.periods)):
            raise ValueError('Equality testing is only allowed for SpaceGroupElements with the same `periods`.')
        if not PointGroupElement.__eq__(self, other):
            return False
        t_diff = self.t - other.t
        t_int = np.linalg.solve(self.periods, t_diff)
        if not allclose(t_int, np.around(t_int)):
            raise ValueError('Pure translation smaller than `periods` detected, make sure `periods` are primitive!')
        return True

    # Need to override hash if eq is changed
    def __hash__(self):
        # U is not hashed, good that we have an integer _R_trf
        R, c, a = self._R_trf, self.conjugate, self.antisymmetry
        return hash((R, c, a))

    def inv(self):
        pg_inv = PointGroupElement.inv(self)
        return SpaceGroupElement(pg_inv, -_mul(pg_inv.R, self.t), self.periods)

    def identity(self):
        """Return identity element with the same structure as self."""
        dim = self.R.shape[0]
        t = ta.zeros((dim,))
        return SpaceGroupElement(PointGroupElement.identity(self), t, self.periods)


class LittleGroupElement(SpaceGroupElement):
    def __init__(self, R, k, t=None, periods=None, conjugate=False, antisymmetry=False, U=None, RSU2=None, phase=None,
                 phase_in_factor=True, _strict_eq=False, *, locals=None):
        """Container for little group elements. The primitive translation vectors of the
        enclosing space group are `periods`, the translation part of this element is `t`.
        Translation part is normalized to within the primitive cell. `k` is measured in
        units of 2pi. `phase` is used to keep track of elements of the covering group."""
        # Allow initialization with a SGE as R, then all other optional parameters are ignored
        ### TODO: initializing with SGE has side effect on SGE, makes it PGE, debug this
        if isinstance(R, SpaceGroupElement):
            if conjugate or antisymmetry or _strict_eq or any(x is not None for x in (t, periods, U, RSU2)):
                raise ValueError('When initializing with a SpaceGroupElement, no optional arguments can be provided except for `phase`.')
            super().__init__(R.R, R.t, R.periods, R.conjugate, R.antisymmetry, R.U, R.RSU2, R._strict_eq, locals=locals)
        # Allow initialization with a PGE as R
        elif isinstance(R, PointGroupElement):
            if t is None or periods is None or conjugate or antisymmetry or _strict_eq or any(x is not None for x in (U, RSU2)):
                raise ValueError('When initializing with a PointGroupElement, must provide `t` and `periods`, '
                                 'but no optional arguments can be provided.')
            super().__init__(R.R, t, periods, R.conjugate, R.antisymmetry, R.U, R.RSU2, R._strict_eq, locals=locals)
        elif not t or not periods:
            raise ValueError('Must provide `t` and `periods`.')
        else:
            super().__init__(R, t, periods, conjugate, antisymmetry, U, RSU2, _strict_eq, locals=locals)
        self.phase = phase

        # Fold t back into the fundamental domain of periods
        self.t = ta.array(self.to_fd(self.t))

        # Make reciprocal lattice vectors
        ### TODO: make it work when there are less translation dimensions than space diimensions

        self.k_periods = np.linalg.inv(self.periods).T
        # Make sure that k is invariant
        k = ta.array(k)
        if not allclose(k, self.to_bz(_mul(self.R, (-1 if self.conjugate else 1) * k))):
            raise ValueError('`k` must be invariant.')
        self.k = ta.array(self.to_bz(k))

        self.phase_in_factor = phase_in_factor

    def to_fd(self, t):
        t_trf = np.linalg.solve(self.periods, t)
        # Make sure that the faces of the FD are treated consistently
        return self.periods @ ((t_trf + 0.5 - 1e-6) % 1 - 0.5 + 1e-6)

    def to_bz(self, k):
        k_trf = np.linalg.solve(self.k_periods, k)
        # Make sure that the faces of the FD are treated consistently
        return self.k_periods @ ((k_trf + 0.5 - 1e-6) % 1 - 0.5 + 1e-6)

    def _factor(self, other):
        if self.phase_in_factor:
            if self.conjugate:
                # antiunitary adds an extra sign to phase factor
                return np.exp(2j*np.pi * _mul(self.k, _mul(self.R, other.t) + other.t))
            else:
                return np.exp(2j*np.pi * _mul(self.k, _mul(self.R, other.t) - other.t))
        else:
            # in this gauge antiunitaries act the same
            t = self.t + _mul(self.R, other.t)
            return np.exp(2j*np.pi * _mul(self.k, t - self.to_fd(t)))

    def factor(self, other):
        """Return the factor in the projective representation corresponding to
        the little group representation, including the k-dependent factor, such that
        d(k, g) * d(k, h) = factor(g, h) * d(k, g*h), where g=self and h=other.
        It is 1 for true representations, including the case when phases are set
        and we store covering group elements."""
        if self.phase is not None and other.phase is not None:
            return 1
        elif self.phase is None and other.phase is None:
            return self._factor(other)
        else:
            raise ValueError('`phase` must be set for both or None for both LittleGroupElements.')

    # Implement multiplication
    def __mul__(self, g2):
        """A LittleGroupElement object g corresponds to the representation by
        d(k, g) = exp(2pi i k @ to_fd(g.t)) * g.U * g.phase,
        where only U is explicitely stored and the other phase factors are implicit. If phase in None,
        then it is taken as 1, and the LittleGroupElements form a projective representation of the
        little group with appropriate phase factors. If it is provided, it is used to generate the
        covering group. To make this a consistent representation, the multiplication rule for phase is:
        (g1 * g2).phase = g1.phase * g2.phase * exp(2pi i k @ (g1.t + g2.t - to_fd(g1.t + g1.R @ g2.t))).
        """
        g1 = self
        if not _eq(g1.k, g2.k):
            raise ValueError('Multiplication is only allowed for LittleGroupElements with the same `k`.')

        res = SpaceGroupElement.__mul__(g1, g2)
        res.t = self.to_fd(res.t)
        res = LittleGroupElement(res, g1.k, phase_in_factor=self.phase_in_factor)

        if g1.phase is None and g2.phase is None:
            res.phase = None
            # Multiplication rule includes an extra phase for U to make projective rep.
            if res.U is not None:
                res.U = res.U * 1/g1.factor(g2)
        elif g1.phase is not None and g2.phase is not None:
            # U parts multiply non-projectively, and we also keep track of overall phase
            if g1.conjugate:
                ### TODO: Double check case with antiunitary symmetries
                res.phase = g1.phase * np.conjugate(g2.phase) * g1._factor(g2)
            else:
                res.phase = g1.phase * g2.phase * g1._factor(g2)
        else:
            raise ValueError('`phase` must be set for both or None for both LittleGroupElements.')

        return res

    # Same as SGE equality, but need extra check that t's can't differ.
    # Check phase if set.
    def __eq__(self, other):
        if not _eq(self.k, other.k):
            raise ValueError('Equality testing is only allowed for LittleGroupElements with the same `k`.')
        if not SpaceGroupElement.__eq__(self, other):
            return False
        if not _eq(self.t, other.t):
            raise ValueError('Pure translation detected, make sure `periods` are primitive!')
        if self.phase is None and other.phase is None:
            return True
        elif self.phase is not None and other.phase is not None:
            return np.abs(self.phase - other.phase) < 1e-6
        else:
            raise ValueError('`phase` must be set for both or None for both LittleGroupElements.')

    def __lt__(self, other):
        if not SpaceGroupElement.__eq__(self, other):
            return SpaceGroupElement.__lt__(self, other)
        else:
            return np.angle(-self.phase - 1e-3j) < np.angle(-other.phase - 1e-3j)

    # Need to override hash if eq is changed
    def __hash__(self):
        # U is not hashed, good that we have an integer _R_trf
        R, c, a = self._R_trf, self.conjugate, self.antisymmetry
        return hash((R, c, a))

    def inv(self):
        inv = LittleGroupElement(SpaceGroupElement.inv(self), self.k, phase_in_factor=self.phase_in_factor)
        inv.t = ta.array(self.to_fd(inv.t))
        factor = inv._factor(self)
        ### TODO: double check antiunitary case
        assert allclose(inv._factor(self), self._factor(inv))
        if self.phase is None:
            inv.phase = None
            if inv.U is not None:
                inv.U = inv.U * factor
        else:
            inv.phase = 1/self.phase * factor
        assert self * inv == self.identity()
        return inv

    def identity(self):
        """Return identity element with the same structure as self."""
        return LittleGroupElement(SpaceGroupElement.identity(self), self.k,
                                  phase=(1 if self.phase else None),
                                  phase_in_factor=self.phase_in_factor)

    def is_phase(self):
        """Return Ture if operator is a pure phase rotation."""
        identity = self.identity()
        return SpaceGroupElement.__eq__(self, identity)



class SpaceGroup(PointGroup):
    def __init__(self, generators, periods=None, double_group=None, _tests=False, tol=1e-9):
        """Class to store space group objects and related representation
        theoretical information. It represents the discrete group generated by
        a set of generators and the translations by `periods`.
        If initialized with a PointGroup or an iterable of PointGroupElements,
        it represents the symmorphic space group with this PointGroup.
        Representation theoretical attributes correspond to the representations
        of its point group. To access advanced space group representation
        information, generate a little_group for a given k."""
        # Implement initialization from a PointGroup
        if isinstance(generators, PointGroup):
            self.generators = set(SpaceGroupElement(g, np.zeros((g.R.shape[0],)), periods) for g in generators)
            self.point_group = generators
            self.periods = np.atleast_2d(periods)
        elif all(isinstance(g, SpaceGroupElement) for g in generators):
            self.periods = next(iter(generators)).periods
            if not all(allclose(g.periods, self.periods) for g in generators):
                raise ValueError('All generators must have the same `periods`.')
            self.generators = generators
            self.point_group = PointGroup(set(PointGroupElement(g.R, g.conjugate, g.antisymmetry,
                                                                g.U, g.RSU2, g._strict_eq)
                                              for g in generators))
        elif all(isinstance(g, PointGroupElement) for g in generators):
            self.generators = set(SpaceGroupElement(g, np.zeros((g.R.shape[0],)), periods) for g in generators)
            self.point_group = PointGroup(generators)
            self.periods = np.atleast_2d(periods)
        else:
            raise ValueError('`generators` must be a PointGroup, or a set of PointGroupElements or SpaceGroupElements.')

        super().__init__(self.generators, double_group=double_group, _tests=_tests, tol=tol)

    def little_group(self, k, phase_in_factor=True):
        # Try fixing the phases, this will raise an error if U's are not set
        try:
            self.fix_U_phases()
        except ValueError:
            pass
        # find group elements that leave k invariant
        ### TODO: brute force for now, could be optimized along the lines of discrete_symmetries
        # Should prefer keeping the minimal_generators if they are part of the LG
        lg = set()
        for g in self.elements:
            try:
                # This fails if k is not invariant under g
                lg.add(LittleGroupElement(g, k=k, phase_in_factor=phase_in_factor))
            except ValueError:
                pass
        return LittleGroup(lg)


class LittleGroup(SpaceGroup):
    def __init__(self, generators, k=None, double_group=None, allow_inconsistent=False, _tests=False, tol=1e-9):
        """Class to store little group objects and related representation
        theoretical information."""
        # Implement initialization from a SpaceGroup
        if isinstance(generators, SpaceGroup):
            self.generators = set(LittleGroupElement(g, k) for g in generators)
            self.periods = next(iter(generators)).periods
            self.k = k
        elif all(isinstance(g, LittleGroupElement) for g in generators):
            self.periods = next(iter(generators)).periods
            self.k = next(iter(generators)).k
            if not all(allclose(g.periods, self.periods) for g in generators):
                raise ValueError('All generators must have the same `periods`.')
            if not all(allclose(g.k, self.k) for g in generators):
                raise ValueError('All generators must have the same `k`.')
            self.generators = generators
        elif all(isinstance(g, SpaceGroupElement) for g in generators):
            self.periods = next(iter(generators)).periods
            if not all(allclose(g.periods, self.periods) for g in generators):
                raise ValueError('All generators must have the same `periods`.')
            self.generators = set(LittleGroupElement(g, k) for g in generators)
            self.k = k
        else:
            raise ValueError('`generators` must be a SpaceGroup, or a set of LittleGroupElements or SpaceGroupElements.')

        super().__init__(self.generators, double_group=double_group, _tests=_tests, tol=tol)

    # Implement consistency checking with the extra exp(i k.t) factors.
    # Maybe unnecessary because fixing for k=0 always results in consistent U's?
    # Only for reps generated from a SG, not always
    @cached_property
    def consistent_U(self):
        if not self.U_set:
            raise ValueError('The U attribute must be set for all goup elements.')

        # Way to retrieve the representative U
        group_dict = {g: g for g in self.elements}
        # Brute force check of full multiplication table
        for g, h in product(self.elements, repeat=2):
            if not allclose(g.factor(h) * group_dict[g * h].U,  g.U @ h.U):
                return False
        else:
            return True

    def fix_U_phases(self):
        pass

    @cached_property
    def decompose_U_rep(self):
        ct = self.character_table_full
        char = np.array([np.trace(g.U) for g in self.unitary_elements_list])
        decomp = ct @ char.conj() / len(self.unitary_elements)
        assert allclose(decomp, np.around(decomp.real))
        return np.around(decomp.real).astype(int)

    def antiunitary_square(self):
        # return the square of the antiunitary generator including all phases
        # only returns a complex phase, we assume TR^2 is a pure phase
        TR = self.antiunitary_generator
        if TR is None:
            raise ValueError('Group contains no antiunitary.')
        TR2 = 1 if TR.conjugate is True else prop_to_id((TR**2).RSU2)[1]
        TR2 = TR2 / TR.factor(TR)
        assert abs(TR2**2 - 1) < 1e-6
        TR2 = int(np.around(TR2).real)
        return TR2

    @cached_property
    def character_table(self):
        r"""
        Return the character table of the unitary part of the group.
        Rows correspond to the different irreps, and columns to the conjugacy
        classes in the order of `self.conjugacy_classes`. As this is a projective
        representation, the phase of the character can differ within a conjugacy
        class, here the characters of `self.class_representatives` are listed.
        """
        return self._character_table(full=False)

    @cached_property
    def character_table_full(self):
        r"""
        Return the character table of the unitary part of the group.
        Rows correspond to the different irreps, and the character is listed
        for all elements in the order of `self.unitary_elements_list`.
        """
        return self._character_table(full=True)

    def _character_table(self, full=False):
        """Return the character table of the unitary part of the group.
        Rows correspond to the different irreps, and columns to the conjugacy
        classes in the order of `self.conjugacy_classes`. As this is a projective
        representation, the phase of the character can differ within a conjugacy
        class, here the characters of `self.class_representatives` are listed.
        If `full`, the character is listed for all elements in the order of
        `self.unitary_elements_list`.

        Notes:
        Character table is generated from the character table of the covering group
        by picking out irreps where the phase factors are represented correctly.
        """
        covering_characters = self.covering_group.character_table

        phase_classes = np.array([(i, g.phase)
                                  for (i, g) in enumerate(self.covering_group.class_representatives)
                                  if g.is_phase()])
        assert all(len(self.covering_group.conjugate_classes[int(i.real)]) == 1 for i, _ in phase_classes)
        characters = np.array([chi for chi in covering_characters
                               if allclose(chi[np.array(np.around(phase_classes[:, 0]).real, dtype=int)],
                                           chi[0] * phase_classes[:, 1])])

        i_cov = []
        for g in self.unitary_elements_list:
            # pick out the class in the covering group corresponding to this element with 1 phase
            g_cov = copy(g)
            g_cov.phase = 1
            assert g_cov in self.covering_group.unitary_elements
            cov_cl = [j for j, cl in enumerate(self.covering_group.conjugate_classes) if g_cov in cl]
            assert len(cov_cl) == 1
            cov_cl = cov_cl[0]
            # assert all([allclose(g.phase, 1) for g in self.covering_group.conjugate_classes[cov_cl]])
            i_cov.append(cov_cl)
        characters_full = characters[:, np.array(i_cov)]
        characters = characters_full[:, np.array([self.unitary_elements_list.index(g) for g in self.class_representatives])]
        # Need to sort them again
        sort_order = np.lexsort(np.round(np.abs(characters.T[::-1] - 1 - 0.1j), 3))
        characters = characters[sort_order, :]
        characters_full = characters_full[sort_order, :]
        class_sizes = np.array([len(c) for c in self.conjugate_classes])
        assert characters.shape[0] == characters.shape[1], characters.shape
        assert allclose(characters @ np.diag(class_sizes) @ characters.T.conj() / sum(class_sizes), np.eye(characters.shape[0]))
        assert allclose(characters_full @ characters_full.T.conj() / sum(class_sizes), np.eye(characters.shape[0]))
        self.character_table = characters
        self.character_table_full = characters_full
        if full:
            return self.character_table_full
        else:
            return self.character_table

    @cached_property
    def covering_group(self):
        """The covering group generated by keeping track of the complex
        phases acquired when multiplying LGE's. It is a finite PointGroup
        containing unitary LittleGroupElements."""
        if all(g.phase is not None for g in self):
            covering_group_ = PointGroup(self)
        else:
            ### TODO: Implement this for general projective representations
            # by keeping track of the U phases.
            cg = set()
            for g in self.unitary_elements:
                assert g.phase is None
                new_g = copy(g)
                new_g.phase = 1
                cg.add(new_g)
            covering_group_ = PointGroup(cg)
        if self._tests:
            covering_group_._tests = True
        return covering_group_

    @cached_property
    def regular_representation(self):
        # Construct the regular representation with permutation matrices for U
        ### TODO: allow sparse matrices for U in PGE to speed this up
        new_generators = set()
        element_dict = {g: i for i, g in enumerate(self.unitary_elements)}
        for g in self.unitary_elements:
            mat = np.zeros((len(self.unitary_elements), len(self.unitary_elements)), dtype=complex)
            for h in self.unitary_elements:
                mat[element_dict[h], element_dict[h*g]] = h.factor(g)
            new_g = copy(g)
            new_g.U = mat
            new_generators.add(new_g)
        reg_rep = type(self)(new_generators)

        if self._tests:
            reg_rep._tests = True
            assert reg_rep.consistent_U
            assert allclose([g.R for g in self.class_representatives], [g.R for g in reg_rep.class_representatives])
            assert allclose([g.t for g in self.class_representatives], [g.t for g in reg_rep.class_representatives])
            assert allclose([g.factor(h) for g, h in product(reg_rep.class_representatives, repeat=2)],
                            [g.factor(h) for g, h in product(self.class_representatives, repeat=2)])
            assert allclose(reg_rep.character_table, self.character_table)
            assert allclose(reg_rep.character_table_full, self.character_table_full)
            assert allclose(reg_rep.decompose_U_rep, reg_rep.character_table[:, 0])
        reg_rep.character_table = self.character_table
        reg_rep.character_table_full = self.character_table_full
        return reg_rep


# -

# ### Tests

# #### Permutation group

# +
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


# -

n = 5
gens = [qsymm.PointGroupElement(p) for p in permutation_generators(n)]
group = qsymm.groups.generate_group(gens)

ct = character_table(group, tol=1e-6)
ct

decompose_representation(group, use_R=True)

# ##### Direct power representations

# this takes a while for large p because the big matrices and qsymm doesn't support sparse U
n = 5
p = 1
p_gens = permutation_generators(n)
reps = power_rep(p_gens, p, sparse=False)
gens = [qsymm.PointGroupElement(p, U=U) for p, U in zip(p_gens, reps)]
group = qsymm.groups.generate_group(gens)

decompose_representation(group, irreps=ct)

# #### Pauli group

# +
sigma = 2 * qsymm.groups.spin_matrices(1/2)
gens = [qsymm.PointGroupElement(np.kron(sigma[2], np.eye(2)).real),
        qsymm.PointGroupElement(np.kron(1j * sigma[1], np.eye(2)).real),
        qsymm.PointGroupElement(np.kron(sigma[0], sigma[0]).real),
        qsymm.PointGroupElement(np.kron(sigma[0], sigma[2]).real)
       ]

pg = PointGroup(gens)
# -

pg.character_table

pg.decompose_R_rep
