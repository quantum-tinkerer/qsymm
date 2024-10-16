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
    e = next(iter(group))
    e = qsymm.groups.identity(dim=e.R.shape[0], shape=e.U.shape[0] if e.U is not None else None)
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
    order = np.argsort(list(map(len, conjugate_classes)))
    conjugate_classes = conjugate_classes[order]
    class_representatives = [min(cl) for cl in conjugate_classes]
    class_by_element = {g: np.argsort(order)[c] for g, c in class_by_element.items()}
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
    norms = character_product(chars, chars, class_sizes)
    chars = np.sqrt(1 / norms[:, None]) * chars
    chars *= np.sign(chars[:, 0])[:, None]
    return chars

def character_product(char1, char2, class_sizes):
    return np.sum(class_sizes[..., :] * char1 * char2.conj(), axis=-1) / sum(class_sizes)

def decompose_representation(group, use_R=False, irreps=None,
                             conjugate_cl=None, class_reps=None, class_by_element=None):
    if any(a is None for a in [conjugate_cl, class_reps, class_by_element]):
        conjugate_cl, class_reps, class_by_element = conjugate_classes(group)
    class_sizes = np.array([len(c) for c in conjugate_cl])
    char = np.array([np.trace(g.R) if use_R else np.trace(g.U) for g in class_reps])
    if irreps is None:
        irreps = character_table(group, tol=1e-6, conjugate_cl=conjugate_cl, class_by_element=class_by_element)
    return character_product(char, irreps, class_sizes)


# from functools import property
from qsymm.groups import generate_group

class PointGroup(set):

    def __init__(self, generators):
    """Class to store point group objects and related representation
    theoretical information. Only supports PointGroupElements."""
        return super().__init__(generators)

    @property
    def elements(self):
        if hasattr(self, '_elements'):
            return self._elements

        self._elements = generate_group(self)
        return self._elements

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

        if any(g.U is None for g in self.elements):
            raise ValueError('The U attribute must be set for all goup elements.')
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


# -

g = qsymm.groups.cubic(tr=False, ph=False, generators=True, spin=1)
pg = PointGroup(g)

pg

# %%time
pg.character_table

pg.class_representatives

# %%time
pg.decompose_R_rep

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
