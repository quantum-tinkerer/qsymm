# +
import numpy as np
import sympy
import qsymm
from itertools import product
from qsymm.linalg import split_list, allclose, commutator
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components

sympy.init_printing(print_builtin=True)
np.set_printoptions(precision=2, suppress=True, linewidth=150)

# %load_ext autoreload
# %autoreload 2

# +
def conjugate_classes(group):
    # make sure the identity is the first class
    e = next(iter(group))
    e = qsymm.groups.identity(dim=e.R.shape[0], shape=e.U.shape if e.U is not None else None)
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
    class_by_element = {g: np.argsort(order)[c] for g, c in class_by_element.items()}
    return conjugate_classes, class_by_element

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
    # assert allclose(np.abs((u1 @ Vh.T[:, ind]).T.conj() @ u2 @ (U.conj()[:, ind])), 1)
    return u1 @ Vh.T[:, ind]

def common_eigenvectors(mats, tol=1e-6):
    eigensubspaces = [np.eye(mats[0].shape[0])]
    for i in range(len(mats)):
        _, new_subspaces = grouped_diag(mats[i], tol=1e-6)
        eigensubspaces = [subspace_intersection(u1, u2, tol=1e-6)
                          for u1, u2 in product(eigensubspaces, new_subspaces)]
        eigensubspaces = [s for s in eigensubspaces if s is not None]
    return eigensubspaces

def character_table(group, tol=1e-6):
    # Using Burnside's method, based on DIXON Numerische Mathematik t0, 446--450 (1967)
    conjugate_c, class_by_elemet = conjugate_classes(group)
    class_sizes = np.array([len(c) for c in conjugate_c])
    M = build_M_matrices(group, conjugate_c, class_by_elemet)
    chars = np.hstack(common_eigenvectors(np.transpose(M, axes=(0, 2, 1)), tol)).T
    norms = np.sum(class_sizes[None, :] * chars * chars.conj(), axis=1)
    chars = np.sqrt(len(group) / norms[:, None]) * chars
    chars *= np.sign(chars[:, 0])[:, None]
    return chars


# -

g = qsymm.groups.cubic(tr=False, ph=False)

len(g)

# %%time
ct = character_table(g, tol=1e-6)

ct.real


