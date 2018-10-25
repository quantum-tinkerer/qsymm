import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import itertools as it
import tinyarray as ta


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def prop_to_id(A):
    # Test if A is proportional to the identity matrix
    # and return the factor as well
    if np.isclose(A[0, 0], 0):
        if np.allclose(A, np.zeros(A.shape)):
            return True, 0
        else:
            return False, 0
    else:
        if np.allclose(A / A[0, 0], np.eye(*A.shape)):
            return (True, A[0, 0])
        else:
            return False, 0


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Alternative to numpy.allclose to test that two ndarrays
    are elementwise close. Unlike the numpy version, the relative
    tolerance is not elementwise, but it is relative to
    the largest entry in the array."""
    a = np.asarray(a)
    b = np.asarray(b)
    # Check if empty arrays, only compare shape
    if a.size == 0:
        return a.shape == b.shape
    atol = atol + rtol * np.max(np.abs(a))
    return np.allclose(a, b, rtol=0, atol=atol, equal_nan=equal_nan)


def mtm(a, B, c):
    # matrix-tensor-matrix multiplication for dimensions 2, 3, 2.
    # Equivalent to 'np.einsum('ij,ajk,kl->ail', a, B, c)'
    # and to 'a @ B @ c', but much faster.
    return np.array([a.dot(b).dot(c) for b in B])


def matrix_basis(dim, traceless=False, antihermitian=False, real=False, sparse=False):
    """"Construct a basis for the vector space of dim x dim matrices,
    that may in addition be traceless, and/or composed of Hermitian
    or skew-Hermitian basis elements.

    Parameters
    ----------
    dim: positive integer
        The dimension of the matrices (dim x dim).
    traceless: boolean
        Whether the vector space only includes traceless matrices (True)
        or not (False).
    antihermitian: boolean
        Whether to use a basis composed of Hermitian matrices (False),
        or skew-Hermitian matrices (True).
    real:
        Whether the basis only spans real matrices. Depending on
        'antihermitian' it spans symmetric (False) or antisymmetric
        (True) matrices only.
    sparse:
        If True, scipy.sparse.csr_matrix is returned, otherwise
        np.ndarray.

    Returns
    ---------
    basis: generator
        A generator that returns matrices that span the vector space.
    """
    if sparse:
        null = lambda: scipy.sparse.lil_matrix((dim, dim), dtype=complex)
        set_type = lambda x: x.tocsr()
    else:
        null = lambda: np.zeros((dim, dim), dtype=complex)
        set_type = lambda x: x
    # Matrix basis for dim x dim matrices. With real coefficients spans
    # Hermitian matrices, with complex spans all matrices
    coeff = (1j if antihermitian else 1)
    # Diagonals
    if real and antihermitian:
        pass
    elif traceless:
        for i in range(dim - 1):
            diag = null()
            diag[i, i] = 1
            diag[dim-1, dim-1] = -1
            yield coeff * set_type(diag)
    else:
        for i in range(dim):
            diag = null()
            diag[i, i] = 1
            yield coeff * set_type(diag)
    # Off-diagonals
    for i, j in it.combinations(list(range(dim)), 2):
        if not (real and antihermitian):
            h = null()
            h[i, j] = 1
            h[j, i] = 1
            yield coeff * set_type(h)
        if not (real and not antihermitian):
            h = null()
            h[i, j] = 1j
            h[j, i] = -1j
            yield coeff * set_type(h)


def family_to_vectors(family, all_keys=None):
    """Convert a list of Model to standard vector representation.

    Parameters
    ----------
    family: iterable of Model objects
        The Model to convert to vector representation.
    all_keys: iterable or None
        Iterable that must contain all the keys in the family.
        An optional parameter that allows to fix the ordering of
        the returned vectors to the same as in all_keys.
        Otherwise, the keys are ordered arbitrarily (all_keys = None).

    Returns
    ----------
    A matrix representing the family, in which the family members form the
    rows."""

    def matrix_to_vector(M):
        M = M.reshape(-1)
        return np.concatenate((M.real, M.imag))

    if all_keys is None:
        all_keys = set()
        for term in family:
            all_keys |= term.keys()
        all_keys = list(all_keys)
    assert all([set(all_keys) >= term.keys() for term in family])
    if not all_keys:
        return np.empty((len(family), 0))
    # Map each family member to a vector, gather the vectors into
    # a matrix in which the vectors form the rows.
    basis_vectors = []
    for member in family:
        vector = [matrix_to_vector(mat) for mat in member.value_list(all_keys)]
        basis_vectors.append(np.hstack(vector))
    return np.vstack(basis_vectors)


def nullspace(A, atol=1e-6, return_complement=False, sparse=False, k_max=-10):
    """Compute an approximate basis for the nullspace of matrix A.

    Parameters:
    -----------
    A : ndarray
        A should be 2-D.
    atol : real
        Absolute tolerance when deciding whether an eigen/singular
        value is zero, so the corresponding vector is included in the
        null space.
    return_complement : bool
        Whether to return the basis of the orthocomplement ot the
        null space as well. Treated as False if sparse is True.
    sparse : bool
        Whether to use sparse linear algebra to find the null space.
    k_max : int
        If k_max<0, the full null-space is found by increasing the number
        of eigenvectors found in each step by abs(k_max).
        If k_max>0, k_max is the maximum number of null space vectors requested.
        If the null space dimension is higher than k_max, basis of a random
        subspace is returned. Can be used to increase performance if the
        maximum null space dimensionality is known.
        Ignored unless sparse is True.

    Returns:
    --------
    ns : ndarray
        Basis of the null space of A as row vectors.
    nsc : ndarray
        Basis of the complement of the null space as row vectors, the
        othonormal basis of the row span of A. Only returned if
        return_complement=True and sparse=False.
    """

    if sparse:
        # Do sparse eigenvalue solving
        # sigma used for shift-invert, should be a small negative value
        sigma = -1e-1
        # Make sure A is sparse matrix
        A = scipy.sparse.csr_matrix(A)
        A.eliminate_zeros()
        # Treat A=0 case
        if np.allclose(A.data, 0, atol=atol/A.shape[1]):
            return np.eye(A.shape[1])
        # Make Hermitian positive definite matrix
        A = A.T.conj().dot(A)

        if k_max > 0:
            # If k_max is specified, find k_max smallest eigenvalues
            evals, evecs = sla.eigsh(A, sigma=sigma, which='LM',
                                     k=k_max, return_eigenvectors=True)
        elif k_max < 0:
            # Successively find more eigenvalues until some not close to zero
            # Number of new null space vectors to find in each step
            k_step = abs(k_max)
            k_max = min(A.shape[0] - 2, k_step)
            while k_max < A.shape[0] - 1:
                evals, evecs = sla.eigsh(A, sigma=sigma, which='LM',
                                         k=k_max, return_eigenvectors=True)
                if np.max(np.abs(evals)) > atol:
                    # We found the first large one
                    break
                else:
                    # All small, need to increase k_max
                    k_max += k_step
            else:
                raise ValueError('A should have at most A.shape[0]-2 dimensional null space.'
                                 'Try using sparse=False.')
        else:
            raise ValueError('k_max must be nonzero.')

        # Only keep eigenvectors with small eigenvalues
        nnz = np.isclose(evals, 0, atol=atol)
        ns = evecs[:, nnz]
        # Orthonormalize null space
        if ns.shape[1] > 0:
            ns, _ = la.qr(ns, mode='economic')
        return ns.T
    else:
        # Do dense SVD
        u, s, vh = la.svd(A, full_matrices = True)
        nnz = np.isclose(s, 0, atol=atol)
        # Make sure it works for arbitrary rectangular matrices
        if len(s) < len(vh):
            nnz = np.concatenate((nnz, [True for _ in range(len(vh) - len(s))]))
        ns = vh[nnz].conj()
        nsc = vh[np.invert(nnz)].conj()
        if return_complement:
            return ns, nsc
        else:
            return ns


def split_list(vals, tol=1e-6):
    # Returns start and end indices of blocks of
    # consecutive close values in a list
    boundaries = np.where(np.abs(np.diff(vals)) > tol/len(vals))[0] + 1
    return np.array([np.insert(boundaries, 0, 0),
                     np.append(boundaries, len(vals))]).T


def simult_diag(mats, tol=1e-6, checks=0):
    """
    Simultaneously diagonalize commuting normal square matrices.
    Recursive algorithm, first diagonalize first matrix, express the
    rest of the matrices in its eigenbasis, and repeat the procedure
    for the rest of the matrices in each of the degenerate eigensubspaces.
    Recursion depth is limited by the number of nontrivial nested
    eigensubspaces, which is at most the number of matrices or
    the size of the matrices.

    Parameters:
    -----------------
    mats : ndarray
        List of commuting matrices, 3 index array.
    tol : float
        Tolerance when deciding degeneracy of eigenvalues.
    checks : int
        Amount of checks to do.
        0: no checks
        1: check the final result
        2: check all the intermediate results

    Returns:
    ---------------
    Ps : list of ndarrays
        List of rectangular matrices of common eigenvectors spanning each of
        the common eigensubspaces. Stacking them gives unitary matrix 'U = np.hstack(Ps)'
        such that 'U.T.conj() @ mats[i] @ U' is diagonal for every 'i'.

    Notes:
    It is recommended to order matrices such that those with eigenvalues that
    are far spaced or exactly degenerate (such as symmetries) are first, and
    those which may have accidental near degeneracies (such as Hamiltonians)
    are last.
    Tolerance is divided by the size of the block when deciding which eigenvalues
    are degenerate. The default value seems to work well, but not extensively
    tested, numerical instabilities are possible.
    """
    def grouped_diag(H, tol=1e-6):
        # Diagonalize normal matrix H and group the eigenvalues and eigenvectors
        # such that approximately equal eigenvalues are grouped together.
        # If H is Hermitian, 'eigh' returns ordered eigenvalues.
        # Returns the grouped eigenvalues, eigenvectors
        Hdag = H.T.conj()
        if allclose(H, Hdag):
            evals, U = la.eigh(H)
        else:
            if not np.allclose(commutator(H, Hdag), 0):
                raise ValueError('Only normal matrix can be diagonalized.')
            evals, U = la.eig(H)
            # Treat complex eigenvalues as 2D vectors
            evvec = np.array([evals.real, evals.imag]).T
            # Find connected clusters of close values
            con = cdist(evvec, evvec) < tol/len(H)
            _, groups = connected_components(con)
            # reorder evals and evecs such that groups are together
            order = np.argsort(groups)
            evals = evals[order]
            U = U[:, order]
            # Round U to unitary using QR, this only mixes vectors
            # from degenerate eigensubspaces
            U, _ = la.qr(U)
        if checks == 2:
            # Check the result
            assert allclose((U.T.conj() @ H @ U), np.diag(evals))
        return evals, U

    # If 1x1 matrix, we are done
    if mats[0].shape == (1, 1):
        return [np.eye(1)]

    # Diagonalize mats[0], if there are no more matrices we are done
    evals, U = grouped_diag(mats[0], tol=tol)
    ind = split_list(evals, tol=tol)
    if len(mats) == 1:
        return [U[:, b:e] for b, e in ind]

    # Check that all matrices commute with mats[0]
    if not np.allclose([commutator(mats[0], mat) for mat in mats[1:]], 0):
        raise ValueError('Only commuting matrices can be simultaneously diagonalized.')

    # Transform the rest of the matrices mats[1:] to the eigenbasis
    # of mats[0] where they are all block-diagonal.
    # Apply the same algorithm to the rest of the matrices in
    # each block recursively.
    matr = mtm(U.T.conj(), mats[1:], U)
    if checks == 2:
        # Check that off-diagonal blocks are small after the transformation
        assert np.all([np.allclose(matr[:, b1:e1, b2:e2], 0)
                for (b1, e1), (b2, e2) in it.combinations(ind, 2)])
    Ps = []
    for b, e in ind:
        P0 = U[:, b:e]
        Pnew = simult_diag(matr[:, b:e, b:e], tol=tol, checks=(2 if checks==2 else 0))
        Ps += [np.dot(P0,P) for P in Pnew]
    if checks > 0:
        # Check the result is diagonal
        U = np.hstack(Ps)
        matsd = mtm(U.T.conj(), mats, U)
        assert np.all([allclose(m, np.diag(np.diagonal(m))) for m in matsd])
    return Ps


def solve_mat_eqn(HL, HR=None, hermitian=False, traceless=False, conjugate=False, sparse=False, k_max=-10):
    """Solve for X the simultaneous matrix equatioins X HL[i] = HR[i] X for every i.
    It is mapped to a system of linear equations, the null space of which gives a basis for
    all sulutions.

    Parameters:
    -----------------
    HL : ndarray or list of ndarrays
        Coefficient matrices of identical square shape, list of arrays of
        shape (n, n) or one array of shape (m, n, n).
    HR : ndarray or list of ndarrays or None
        Same as HL. If HR = None, HR = HL is used.
    hermitian, traceless : booleans
        If X has to be hermitian, use hermitian = True.
        If X has to be traceless, use traceless = True.
    conjugate : boolean or list of booleans
        If True, solve X HL[i] = HR[i] X^* instead.
        If a list with the same length as HL and HR is provided, conjugation
        is applied to the equations with index corresponding to the True entries.
    sparse : bool
        Whether to use sparse linear algebra to find the solutions.
    k_max : int
        If k_max<0, all solutions are found by increasing the number
        of soulutions sought in each step by abs(k_max).
        If k_max>0, k_max is the maximum number of solutions requested.
        If the solution space dimension is higher than k_max, basis of a random
        subspace is returned. Can be used to increase performance if the
        maximum number of solutions is known.
        Ignored unless sparse is True.

    Returns:
    ---------------
    ndarray of shape (l, n, n), list of linearly independent square matrices
        that span all solutions of the eaquations.
    """
    HL = np.array(HL)
    if HR is None:
        HR = HL
    else:
        HR = np.array(HR)
    if HL.shape != HR.shape:
        raise ValueError('HL and HR must have the same shape')
    if isinstance(conjugate, bool):
        conjugate = [conjugate] * len(HL)
    if len(conjugate) != len(HL):
        raise ValueError('conugate must have the same length as HL')
    if len(HL.shape) == 3:
        if HL.shape[1] != HL.shape[2]:
            raise ValueError('HL and HR must be (a list of) square matrices')
    else:
        raise ValueError('HL and HR must have the shape (n, n) or (m, n, n)')

    dim = HL.shape[-1]
    # number of basis matrices
    N = dim**2 - (1 if traceless else 0)
    if N == 0:
        return np.empty((0, dim, dim))

    # Prepare for differences in sparse and dense algebra
    if sparse:
        # It is worth doing sparse multiplication if the matrices are
        # over 100 x 100
        if HL.shape[-1] >= 100:
            HL = [scipy.sparse.csr_matrix(hL) for hL in HL]
            HR = [scipy.sparse.csr_matrix(hR) for hR in HR]
            basis = lambda: matrix_basis(dim, traceless=traceless, sparse=True)
        else:
            basis = lambda: matrix_basis(dim, traceless=traceless, sparse=False)
        vstack = scipy.sparse.vstack
        bmat = scipy.sparse.bmat
        # Cast it to coo format and reshape
        flatten = lambda x: scipy.sparse.coo_matrix(x).reshape((x.shape[0]*x.shape[1], 1))
    else:
        basis = lambda: matrix_basis(dim, traceless=traceless, sparse=False)
        vstack = np.vstack
        bmat = np.block
        flatten = lambda x: x.reshape((-1, 1))

    # Calculate coefficients from commutators of the basis matrices
    null_mat = []
    for hL, hR, conj in zip(HL, HR, conjugate):
        if conj:
            row = [flatten(mat.dot(hL) - hR.dot(mat.conj())) for mat in basis()]
        else:
            row = [flatten(mat.dot(hL) - hR.dot(mat)) for mat in basis()]
        null_mat.append(row)

    null_mat = bmat(null_mat)

    if hermitian:
        # Simultaneaous equations for real and immaginary parts
        # Lapack guarantees that the SVD of a real matrix is real
        null_mat = vstack((null_mat.real, null_mat.imag))

    # Find the null space of null_mat
    ns = nullspace(null_mat, sparse=sparse, k_max=k_max)
    # Make all solutions
    # 'ij,jkl->ikl'
    basis = lambda: matrix_basis(dim, traceless=traceless, sparse=False)
    return np.array([sum((v * mat for v, mat in zip(vec, basis()))) for vec in ns])


def rref(m, rtol = 1e-3, return_S=False):
    """ Bring a matrix to reduced row echelon form.

    Parameters
    ----------
    m: 2D numpy.ndarray
        The matrix on which to perform row reduction.
    rtol: float
        The tolerance relative to the largest matrix element
        for what is considered a zero entry.
    return_S: bool
        Whether to return the matrix of linear operations
        that brings the input matrix to reduced row echelon form.

    Returns
    ---------
    red: 2D numpy.ndarray
        The input matrix in reduced row echelon form.
    S: 2D numpy.ndarray
        The matrix of operations that brings the input matrix
        to reduced row echelon form. Only returned if return_S = True.

    Note: If r is the reduced row echelon form of the input matrix m, then
        m = S.dot(r).
    """

    red = np.array(m)
    rows, columns = red.shape

    # Define row operations
    def pivot(i, j):
        P = np.eye(rows, dtype=complex)
        P[i, i] = P[j, j] = 0
        P[i, j] = P[j, i] = 1
        return P

    def rowsub(i, coeffs):
        P = np.eye(rows, dtype=complex)
        P[:, i] -= coeffs
        P[i, i] = 1
        return P

    def rowmul(i, coeff):
        P = np.eye(rows, dtype=complex)
        P[i, i] = coeff
        return P

    # Stop if m is identically zero
    if np.allclose(red, 0):
        return red

    S = np.eye(rows, dtype=complex)
    tol = np.max(np.abs(m)) * rtol

    lead = 0
    for r in range(rows):
        # find leading value below row r in lead column
        while lead < columns:
            i = np.argmax(np.abs(red[r:, lead])) + r
            lv = red[i, lead]
            if np.abs(lv) < tol:
                lead += 1
                continue
            else:
                break
        else:
            break

        T = pivot(i, r)
        S = T.dot(S)
        red = T.dot(red)
        T = rowmul(r, 1/lv)
        S = T.dot(S)
        red = T.dot(red)
        T = rowsub(r, red[:, lead])
        S = T.dot(S)
        red = T.dot(red)
        lead += 1
    if return_S:
        return red, S
    else:
        return red


def sparse_basis(bas, num_digits=3, reals=False):
    """Reduce the number of nonzero entries in a (basis) matrix by
    rounding the matrix and bringing it to reduced row echelon form.

    Parameters
    ----------
    bas: numpy.ndarray
        The (basis) matrix to prettify and round.
    num_digits: positive integer
        Number of significant digits to which to round.
    reals: boolean
        If True, the real and imaginary parts of the matrix
        are treated separately.

    Returns:
    ----------
    A rounded, sparsified matrix with the same row span as the input matrix.

    This is an attempt at matrix sparsification, i.e. we attempt to construct
    the sparsest matrix that has the same row rank. Note that the reduction to
    row echelon form is numerically unstable, so this function should be used
    with caution. """

    if len(bas) < 1:
        return bas
    if reals:
        re = np.real(bas)
        im = np.imag(bas)
        bas = np.hstack((re, im))
    bas_mat = rref(bas, rtol=10**(-num_digits))
    if reals:
        re, im = np.split(bas_mat, 2, axis=1)
        bas_mat = re + 1j * im
    # Round to num_digits and return only nonvanishing rows
    bas_mat = np.round(bas_mat, num_digits)
    bas_mat = np.vstack([row for row in bas_mat if not np.allclose(row, 0, atol=10**(-num_digits))])
    if len(bas_mat) < len(bas):
        warnings.warn('Removed linearly dependent terms from the family during sparsification. '+ \
                      'Resulting family will contain fewer members.')
    return np.round(bas_mat, num_digits)


def _inv_int(A):
    """Invert an integer square matrix A. """
    _A = ta.array(A, int)
    if A == np.empty((0, 0)):
        return A
    if _A != A or abs(la.det(A)) != 1:
        raise ValueError('Input needs to be an invertible integer matrix')
    return ta.array(np.round(la.inv(_A)), int)
