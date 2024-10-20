# +
import numpy as np
import sympy
import qsymm
from itertools import product
from qsymm.linalg import split_list, allclose, commutator
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import scipy.sparse as scsp

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
        g = rest.pop()
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
    class_reps = [next(iter(c)) for c in conjugate_classes]
    for x, y in product(group, repeat=2):
        z = x * y
        if z in class_reps:
            M[class_by_elemet[x], class_by_elemet[y], class_by_elemet[z]] +=1
    return M

def grouped_diag(H, tol=1e-6):
    # Group the eigenvalues and eigenvectors of matrix H
    # such that approximately equal eigenvalues are grouped together.
    # Returns the grouped eigenvalues, eigenvectors
    evals, U = np.linalg.eig(H)
    # Treat complex eigenvalues as 2D vectors
    evvec = np.array([evals.real, evals.imag]).T
    # Find connected clusters of close values
    con = cdist(evvec, evvec) < tol/len(H)
    _, groups = connected_components(con)
    U = [np.linalg.qr(U[:, groups == i])[0] for i in range(max(groups+1))]
    evals = np.array([evals[groups == i][0] for i in range(max(groups+1))])
    return evals, U

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
    assert allclose(A_reduced @ A_reduced.T.conj(), np.eye(A_reduced.shape[0])), (U, S, Vh)
    return u1 @ Vh.T.conj()[:, ind]

def common_eigenvectors(mats, tol=1e-6, quit_when_1d=False):
    eigensubspaces = [np.eye(mats[0].shape[0])]
    for i in range(len(mats)):
        _, new_subspaces = grouped_diag(mats[i], tol=1e-6)
        eigensubspaces = [subspace_intersection(u1, u2, tol=1e-6)
                          for u1, u2 in product(eigensubspaces, new_subspaces)]
        eigensubspaces = [s for s in eigensubspaces if s is not None]
        if all(s.shape[1] == 1 for s in eigensubspaces) and quit_when_1d:
            break
    return eigensubspaces

def character_table(group, tol=1e-6, conjugate_cl=None, class_by_element=None):
    # Using Burnside's method, based on DIXON Numerische Mathematik t0, 446--450 (1967)
    if conjugate_cl is None or class_by_element is None:
        conjugate_cl, _, class_by_element = conjugate_classes(group)
    class_sizes = np.array([len(c) for c in conjugate_cl])
    M = build_M_matrices(group, conjugate_cl, class_by_element)
    chars = np.hstack(common_eigenvectors(np.transpose(M, axes=(0, 2, 1)),
                                          tol, quit_when_1d=True)).T
    norms = character_product(chars, chars, class_sizes, check_int=False)
    chars = np.sqrt(1 / norms[:, None]) * chars
    chars *= np.sign(chars[:, 0])[:, None]
    return chars

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

    def __init__(self, generators, double_group=None):
        """Class to store point group objects and related representation
        theoretical information. Only supports PointGroupElements.
        It represents the discrete group generated by a set of generators,
        these can be accessed through a set interface."""

        super().__init__(generators)
        self.generators = generators

        if not all(isinstance(g, PointGroupElement) for g in generators):
            raise ValueError('Only iterables of PointGroupElements is supported.')
        if not all(g.conjugate is False for g in generators):
            raise NotImplementedError('Only unitary (anti)symmetries are supported.')

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

    @property
    def elements(self):
        if hasattr(self, '_elements'):
            return self._elements

        self._elements = generate_group(self)
        return self._elements

    @property
    def minimal_generators(self):
        if hasattr(self, '_minimal_generators'):
            return self._minimal_generators

        minimal_generators = find_generators(self.elements)
        if len(minimal_generators) >= len(self.generators):
            self._minimal_generators = self.generators
        else:
            self._minimal_generators = minimal_generators
        return self._minimal_generators

    @property
    def consistent_U(self):
        if hasattr(self, '_consistent_U'):
            return self._consistent_U

        if not self.U_set:
            raise ValueError('The U attribute must be set for all goup elements.')
        self._consistent_U = check_U_consistency(self.elements)
        return self._consistent_U

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
            print('-', end='')
            new_gens = []
            for (g, n, phase, sign), m in zip(gens_orders, ms):
                g_new = copy(g)
                g_new.U = g_new.U * np.exp(1j * (-phase/n + 2 * np.pi / n * (m - (sign-1)/4)))
                new_gens.append(g_new)
            if check_U_consistency(generate_group(new_gens)):
                self._minimal_generators = new_gens
                self._elements = generate_group(self.minimal_generators)
                self.generators = {g for g in self._elements if g in self}
                ### TODO: is there a more elegant way to update set contents?
                super().__init__(self.generators)
                self._consistent_U = True
                break
        else:
            raise ValueError('Phase fixing failed! Try changing the `double_group` setting.')

    def _set_conjugate_classes(self):
        (self._conjugate_classes,
         self._class_representatives,
         self._class_by_element) = conjugate_classes(self.elements)

    @property
    def conjugate_classes(self):
        if hasattr(self, '_conjugate_classes'):
            return self._conjugate_classes

        self._set_conjugate_classes()
        return self._conjugate_classes

    @property
    def class_by_element(self):
        if hasattr(self, '_class_by_element'):
            return self._class_by_element

        self._set_conjugate_classes()
        return self._class_by_element

    @property
    def class_representatives(self):
        if hasattr(self, '_class_representatives'):
            return self._class_representatives

        self._set_conjugate_classes()
        return self._class_representatives

    @property
    def character_table(self):
        if hasattr(self, '_character_table'):
            return self._character_table

        self._character_table = character_table(self.elements, self.conjugate_classes, self.class_by_element)
        return self._character_table

    @property
    def decompose_U_rep(self):
        if hasattr(self, '_decompose_U_rep'):
            return self._decompose_U_rep

        if not self.consistent_U:
            self.fix_U_phases()
        self._decompose_U_rep = decompose_representation(self.elements, use_R=False,
                                                         irreps=self.character_table,
                                                         conjugate_cl=self.conjugate_classes,
                                                         class_reps=self.class_representatives,
                                                         class_by_element=self.class_by_element)
        return self._decompose_U_rep

    @property
    def decompose_R_rep(self):
        if hasattr(self, '_decompose_R_rep'):
            return self._decompose_R_rep

        self._decompose_R_rep = decompose_representation(self.elements, use_R=True,
                                                         irreps=self.character_table,
                                                         conjugate_cl=self.conjugate_classes,
                                                         class_reps=self.class_representatives,
                                                         class_by_element=self.class_by_element)
        return self._decompose_R_rep

    def symmetry_adapted_basis(self, tol=1e-9):
        """Find a symmetry adapted basis of the representation in U.
        Returns a list of sets of basis vectors, each set spanning an
        invariant subspace. The ordering corresponds to the order
        nonzero weight irreps appear in `decompose_U_rep`. The division
        of subspaces belonging to the same irrep is not unique."""
        bases = []
        for chi, n in zip(self.character_table, self.decompose_U_rep):
            if n == 0:
                continue
            basis_chi = np.empty((0, self.U_shape[0]))
            basis_rank = 0
            m = 0
            for v in np.eye(self.U_shape[0]):
                # project out already found subspaces
                if bases:
                    v = v - np.hstack(bases) @ (np.hstack(bases).T.conj() @ v)
                w = np.sum([chi[self.class_by_element[g]] * g.U @ v for g in self.elements], axis=0)
                new_rank = np.linalg.matrix_rank(np.vstack([basis_chi, [w]]), tol)
                if new_rank > basis_rank:
                    basis_chi = np.vstack([basis_chi, [w]])
                    basis_rank = new_rank
                if allclose(basis_rank, chi[0]):
                    bases.append(np.linalg.qr(basis_chi.T)[0])
                    basis_rank = 0
                    basis_chi = np.empty((0, self.U_shape[0]))
                    m += 1
                    if m == n:
                        break
        return bases


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
        self._R_trf = ta.array(np.around(periods @ self.R @ periods.T), int)
        if not allclose(self.R, self._R_trf):
            raise ValueError('Rotation is incompatible with lattice periods.')
        self.t = ta.array(t)

    # Implement multiplication
    def __mul__(self, g2):
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
                 _strict_eq=False, *, locals=None):
        """Container for little group elements. The primitive translation vectors of the
        enclosing space group are `periods`, the translation part of this element is `t`.
        Translation part is normalized to within the primitive cell. `k` is measured in
        units of 2pi. `phase` is used to keep track of elements of the covering group."""
        # Allow initialization with a SGE as R, then all other optional parameters are ignored
        ### TODO: initializing with SGE has side effect on SGE, makes it PGE, debug this
        if isinstance(R, SpaceGroupElement):
            if conjugate or antisymmetry or _strict_eq or any(x is not None for x in (t, periods, U, RSU2)):
                raise ValueError('When initializing with a SpaceGroupElement, no optional arguments can be provided.')
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
        if not allclose(k, self.to_bz(_mul(self.R, k))):
            raise ValueError('`k` must be invariant under `R`.')
        self.k = ta.array(self.to_bz(k))

    def to_fd(self, t):
        t_trf = np.linalg.solve(self.periods, t)
        # Make sure that the faces of the FD are treated consistently
        return self.periods @ ((t_trf + 0.5 - 1e-6) % 1 - 0.5 + 1e-6)

    def to_bz(self, k):
        k_trf = np.linalg.solve(self.k_periods, k)
        # Make sure that the faces of the FD are treated consistently
        return self.k_periods @ ((k_trf + 0.5 - 1e-6) % 1 - 0.5 + 1e-6)

    # Implement multiplication
    def __mul__(self, g2):
        """A LittleGroupElement object g corresponds to the representation by
        exp(2pi i k @ to_fd(g.t)) * g.U * g.phase,
        where only U is explicitely stored and the other phase factors are implicit. If phase in None,
        then it is taken as 1, and the LittleGroupElements form a projective representation of the
        little group with appropriate phase factors. If it is provided, it is used to generate the
        covering group. To make this a consistent representation, the multiplication rule for phase is:
        (g1 * g2).phase = g1.phase * g2.phase * exp(2pi i k @ (g1.t + g2.t - to_fd(g1.t + g1.R @ g2.t))).
        """
        g1 = self
        if not _eq(g1.k, g2.k):
            raise ValueError('Multiplication is only allowed for LittleGroupElements with the same `k`.')
        if g1.phase is None and g2.phase is None:
            # Same multiplication rule as SGE, except t is pulled back to FD
            t = self.to_fd(g1.t + _mul(g1.R, g2.t))
            return LittleGroupElement(PointGroupElement.__mul__(g1, g2), g1.k, t, g1.periods)
        elif g1.phase is not None and g2.phase is not None:
            ### TODO: Add case with antiunitary symmetries
            t = self.to_fd(g1.t + _mul(g1.R, g2.t))
            phase = g1.phase * g2.phase * np.exp(2j*np.pi * _mul(self.k, g1.t + g2.t - t))
            return LittleGroupElement(PointGroupElement.__mul__(g1, g2), g1.k, t, g1.periods, phase=phase)
        else:
            raise ValueError('`phase` must be set for both or None for both LittleGroupElements.')

    # Same as SGE equality, but need extra check that t's can't differ.
    # Check phase if set.
    def __eq__(self, other):
        if not _eq(self.k, other.k):
            raise ValueError('Equality testing is only allowed for LittleGroupElements with the same `k`.')
        if not SpaceGroupElement.__eq__(self, other):
            return False
        if not _eq(self.t, other.t):
            raise ValueError('Pure translation detected, make sure `periods` are primitive!')
        if g1.phase is None and g2.phase is None:
            return True
        elif g1.phase is not None and g2.phase is not None:
            return _eq(g1.phase, g2.phase)
        else:
            raise ValueError('`phase` must be set for both or None for both LittleGroupElements.')

    # Need to override hash if eq is changed
    def __hash__(self):
        # U is not hashed, good that we have an integer _R_trf
        R, c, a = self._R_trf, self.conjugate, self.antisymmetry
        return hash((R, c, a))

    def inv(self):
        sg_inv = SpaceGroupElement.inv(self)
        sg_inv.t = self.to_fd(sg_inv.t)
        if self.phase is None:
            phase = None
        else:
            phase = 1/self.phase * np.exp(-2j*np.pi * _mul(self.k, self.t + sg_inv.t))
        return LittleGroupElement(sg_inv, self.k, phase=phase)

    def identity(self):
        """Return identity element with the same structure as self."""
        return LittleGroupElement(SpaceGroupElement.identity(self), self.k, phase=self.phase)


class SpaceGroup(PointGroup):
    def __init__(self, generators, periods, double_group=None):
        # Implement initialization from a PointGroup
        pass

    def little_group(self, k):
        pass

class LittleGroup(SpaceGroup):
    def __init__(self, generators, periods, k, double_group=None):
        pass


# -

model = qsymm.Model('k_x * sigma_x + k_y * sigma_y + k_z * sigma_z')
R1 = qsymm.groups.rotation(1/4, [0, 0, 1])
R2 = qsymm.groups.rotation(1/2, [1, 0, 0])
R2.apply(R1.apply(model)) - (R2 * R1).apply(model)

S1 = SpaceGroupElement(R1, t=[0, 0, 1/4], periods=np.eye(3))
S2 = SpaceGroupElement(R2, t=[0, 0, 0], periods=np.eye(3))
SG = qsymm.groups.generate_group([S1, S2])

len(SG)

# %%time
(S1 * S2).t

(S1 * S1**(-1)).t

(S1**(-1)).t

S1 == S2

L1 = LittleGroupElement(S1, k=[0, 0, 1/2], phase=1)
L2 = LittleGroupElement(S2, k=[0, 0, 1/2], phase=1)

L1 * L2

(L1**4).phase

(L1 * L1.inv()).phase

L1 == L2

type(S1)

g = qsymm.groups.cubic(tr=False, ph=False, generators=True, spin=0, double_group=False)
g = [PointGroupElement(h.R, U=np.kron(np.eye(2), h.U), RSU2=h.RSU2) for h in g]
pg = PointGroup(g)

pg.decompose_U_rep

pg.symmetry_adapted_basis()

g = qsymm.groups.cubic(tr=False, ph=False, generators=True, spin=1/2, double_group=True)
g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
pg = PointGroup(g)
pg

pg.fix_U_phases()

g = qsymm.groups.cubic(tr=False, ph=False, generators=True, spin=1, double_group=False)
g = [PointGroupElement(h.R, U=np.exp(2j * np.pi * np.random.random()) * h.U, RSU2=h.RSU2) for h in g]
pg = PointGroup(g)

check_U_consistency(pg.elements)

pg.consistent_U

pg.minimal_generators

pg.fix_U_phases()

check_U_consistency(pg.elements)

check_U_consistency(pg)

# %%time
pg.character_table.real

pg.class_representatives

# %%time
pg.decompose_U_rep

# ### Tests

# #### Cubic group

g = qsymm.groups.cubic(tr=False, ph=False)

len(g)

# %%time
ct = character_table(g, tol=1e-6)

ct.real


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

# this takes a while because the big matrices and qsymm doesn't support sparse U
n = 5
p = 4
p_gens = permutation_generators(n)
reps = power_rep(p_gens, p, sparse=False)
gens = [qsymm.PointGroupElement(p, U=U) for p, U in zip(p_gens, reps)]
group = qsymm.groups.generate_group(gens)

decompose_representation(group, irreps=ct)
