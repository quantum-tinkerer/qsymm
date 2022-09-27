from .symmetry_finder import _reduce_hamiltonian
from .linalg import simult_diag, mtm, allclose, prop_to_id, split_list
from .groups import PointGroupElement


def takagi(A):
    """Takagi decomposition of symmetric matrix A as A = U D U^T
    where D>=0 diagonal and U is unitary. The column vectors x of U
    are the coneigenvalues of A satisfying A x^* = d * x with d>0
    d, which is the corresponding diagonal element in D.
    See: Applied Mathematics and Computation 234 (2014) 380–384

    Parameters
    ----------
    A : ndarray
        symmetric (complex) matrix

    Returns
    -------
    D : ndarray
        Vector of the coneigenvalues of A in decreasing order.
    U : ndarray
        Array of coneigenvectors of A.
    """
    if not np.allclose(A, A.T):
        raise ValueError('A has to be symmetric.')
    if np.allclose(A @ A.T.conj(), np.eye(A.shape[0])):
        # Skip SVD for unitary matrix
        U = A
        Z = U.T.conj()
    else:
        U, s, Vh = la.svd(A)
        Z = U.T.conj() @ Vh.T
    # Need to take safe square root of Z, the branch cut should not go
    # through any eigenvalues.
    vals, vecs = la.schur(Z)
    vals = np.diag(vals)
    # Find largest gap between eigenvalues
    phases = np.sort(np.angle(vals))
    dph = np.append(np.diff(phases), phases[0] + 2*np.pi - phases[-1])
    i = np.argmax(dph)
    shift = -np.pi - (phases[i] + dph[i]/2)
    # Take matrix square root with branch cut in largest gap
    vals = np.sqrt(vals * np.exp(1j * shift)) * np.exp(-0.5j * shift)
    sqrtZ = vecs @ np.diag(vals) @ vecs.T.conj()
    Uz = U @ sqrtZ
    D = Uz.T.conj() @ A @ Uz.conj()
    assert np.allclose(D - np.diag(np.diag(np.real(D))), 0)
    D = np.diag(np.real(D))
    return D, Uz
    
def youla_antisymm(A):
    """Youla decomposition of antisymmetric matrix A as A = U S U^T
    where S is real antisymmetric block-diagonal with 1x1 or 2x2 blocks
    and U is unitary. The 2x2 blocks have the structure [[0, d], [-d, 0]]
    with d > 0 the singular values of A in decreasing order. The 1x1
    blocks are zero and are at the end. Two consecutive column vectors of U
    span a coneigensubspace of A. If A is real, U is real.
    See: Linear Algebra and its Applications, Volume 422, Issue 1, Pages 29-38

    Parameters
    ----------
    A : ndarray
        Antisymmetric (complex) matrix.

    Returns
    -------
    S : ndarray
        Youla normal form of A.
    U : ndarray
        Array of coneigenvectors of A.

    Note
    ----
    In the worst case scenario (when QR on the full U is carried out in every
    iteration) the algorithm scales as n^4. For yet unknown reasons the
    scaling seems to be between n^3 and n^4 in all cases.
    """
    if not np.allclose(A, -A.T):
        raise ValueError('A has to be antisymmetric.')
    if np.allclose(A @ A.T.conj(), np.eye(A.shape[0])):
        # Skip SVD for unitary matrix
        U = A.copy()
    else:
        U, _, _ = la.svd(A)
    vecs = np.empty((U.shape[0], 0))
    while np.any(U):
        # Pick the first vector x and generate image y. y is orthogonal to x.
        x = U[:, [0]]
        y = - A @ x.conj()
        if U.shape[1] <= 1 or np.allclose(y, 0):
            # we reached the zero singular values
            vecs = np.hstack([vecs, U])
            break
        # Remove x
        U = U[:, 1:]
        y = y / la.norm(y)
        # Find and delete the vector from U that has the largest
        # overlap with y
        overlaps = np.abs(y.T.conj() @ U)[0]
        i = np.argmax(overlaps)
        U = np.delete(U, i, 1)
        # If y wasn't exactly a vector in U,
        # orthogonalize the rest of the vectors to y
        if not np.isclose(overlaps[i], 1):
            overlaps = np.delete(overlaps, i)
            # Only vectors with degenerate singular values can have overlap with y,
            # only orthogonalize the subspace with nonzero overlap
            subspace = np.logical_not(np.isclose(overlaps, 0))
            Q, _ = la.qr(np.hstack([y, U[:, subspace]]), mode='economic')
            # The vectors left in U are all orthogonal to
            # x and y and are singular vectors of A.
            U[:, subspace] = Q[:, 1:]
        # Attach x and y to the new basis, by construction they form a
        # block proportional to i * sigma_y.
        vecs = np.hstack([vecs, x, y])
    # if not np.isclose(np.abs(la.det(vecs)), 1):
    vecs, _ = la.qr(vecs)  
    S = vecs.T.conj() @ A @ vecs.conj()
    # Check the result
    atr = S.copy()
    for i in range(len(A) // 2):
        assert np.isclose(atr[2*i, 2*i+1], -atr[2*i+1, 2*i])
        atr[2*i, 2*i+1], atr[2*i+1, 2*i] = 0, 0
    assert np.allclose(atr, 0), (np.unravel_index(np.argmax(np.abs(atr)), atr.shape),
                                 np.max(np.abs(atr)))
    return S, vecs

def youla(A, rtol=1e-6):
    """Youla decomposition of connormal matrix A as A = U S U^T
    where S is real block-diagonal with 1x1 or 2x2 blocks and U is unitary.
    The 2x2 blocks have the structure [[d, e], [-e, d]] with d, e > 0.
    The blocks are ordered with non-increasing diagonal values.
    Column vectors of U belonging to a block span a coneigensubspace of A.
    See: Linear Algebra and its Applications, Volume 422, Issue 1, Pages 29-38

    Parameters
    ----------
    A : ndarray
        Connormal matrix satisfying A^+ A = A^* A^T.
    rtol : float
        relative tolerance when splitting degenerate blocks.

    Returns
    -------
    S : ndarray
        Youla normal form of A.
    U : ndarray
        Array of coneigenvectors of A.
    """
    if not np.allclose(A.T.conj() @ A, A.conj() @ A.T):
        raise ValueError('Only connormal matrices can be brought to Youla normal form.')
    # Form symmetric and antisymmetric parts of A
    S = 1/2 * (A + A.T)
    K = 1/2 * (A - A.T)
    if np.allclose(S, 0):
        # If S vanishes, do antisymmetric Youla.
        return youla_antisymm(K)
    else:
        # Diagonalize S by Takagi decomposition
        D, U = takagi(S)
    if np.allclose(K, 0):
        # If K vanishes, we are done.
        return np.diag(D), U
    # Transform to this basis, it is block-diagonal and real with 
    # blocks corresponding to degenerate blocks in D.
    K = U.T.conj() @ K @ U.conj()
    assert np.allclose(K.imag, 0), (K, A)
    assert np.allclose(K, -K.T), (K, A)
    tol = np.max(D)*rtol/len(D)
    blocks = split_list(D, tol=tol)
    U2s = []
    Sigmas = []
    for b, e in blocks:
        Sigma, U2 = youla_antisymm(K[b:e, b:e].real)
        U2s.append(U2)
        Sigmas.append(Sigma)
        assert np.allclose(Sigma.imag, 0)
        assert np.allclose(U2.imag, 0)
        assert np.allclose(np.diag(D[b:e]), U2.T.conj() @ np.diag(D[b:e]) @ U2.conj(), atol=tol*(e-b))
    U2 = la.block_diag(*U2s)
    Sigma = la.block_diag(*Sigmas)
    D = np.diag(D)
    Sigma += D
    U = U @ U2
    # Check result
    atr = U.T.conj() @ A @ U.conj()
    assert np.allclose(Sigma, atr, atol=tol*len(D)), 'Numerical instability encountered, '\
           + 'maximum deviation in result: {}'.format(np.max(Sigma - atr))\
           + '\n Try increasing rtol. atol = {}.'.format(tol*len(D))
    return Sigma, U

def decompose_tensor(A, unitary=False):
    # Decompose tensor product A_abnm = B_ab * C_nm
    # If unitary=True, if A is unitary also make B and C unitary.
    sh = A.shape
    # print(sh)
    assert len(sh) == 4
    for a, b in it.product(range(sh[0]), range(sh[1])):
        C = A[a, b, :, :]
        if not np.allclose(C, 0):
            break
    else:
        # If all blocks are zero, it's the tensor product of zeros
        return np.zeros(sh[0:2]), np.zeros(sh[2:4])
    for n, m in it.product(range(sh[2]), range(sh[3])):
        B = A[:, :, n, m]
        if not np.allclose(B, 0):
            B = B / B[a, b]
            break
    assert np.allclose(np.einsum('ab, nm-> abnm', B, C), A), \
            'A is not a tensor product'
    if unitary:
        Bp, Bcoeff = prop_to_id(B @ B.T.conj()) 
        Cp, Ccoeff = prop_to_id(C @ C.T.conj())
        assert Bp and Cp
        B = B / np.sqrt(Bcoeff)
        C = C / np.sqrt(Ccoeff)
    return B, C


def pg_symmetry_adapted_basis(symmetries):
    """Find the symmetry adapted basis of discrete symmetry group.
    The real space rotation part of the PointGroupElements is 
    ignored."""
    symmetries = symmetries
    # Unitary symmetries
    Usym = [g for g in symmetries if not g.conjugate]
    if len(Usym) == 0:
        Usym.append(symmetries[0].identity())
    # Antiunitary symmetries
    Asym = [g for g in symmetries if g.conjugate]
    # Find symmetry adapted basis for unitary part
    Us = np.array([g.U for g in Usym])
    Preduced = _reduce_hamiltonian(Us)
    # Further merge irreps if there are antiunitaries
    for TR in Asym:
        Preduced = antiunitary_symmetry_adapted_basis(Preduced, TR)
    return Preduced


def antiunitary_symmetry_adapted_basis(Ps, TR):
    """Calculate symmetry adapted basis from a symmetry adapted basis
    Ps after adding an antiunitary operator TR. All irreps in Ps remain
    closed, only identical irreps are linearly combined and irreps merged
    pairwise if TR connects them."""
    assert TR.conjugate
    UTR = TR.U
    Preduced = ()
    Ps = list(Ps)
    blocks = list(range(len(Ps)))
    while blocks:
        i = blocks[0]
        P1 = Ps[i]
        # Find the block TR connects P1 to
        for j in blocks:
            if not np.allclose(np.hstack(P1).T.conj() @ UTR @ np.hstack(Ps[j]).conj(), 0):
                break
        P2 = Ps[j]
        blocks.remove(i)
        # It must be a tensor product, U acts within irreps, V mixes irreps
        U, V = decompose_tensor(np.einsum('nja, jk, mkb-> abnm', P1.conj(), UTR, P2.conj()), unitary=True)
        # print(U, V)
        if i == j:
            # TR mixes identical irreps.
            # Find unitaries, such that an 'otrthogonal' transformations
            # brings U and V into canonical from.
            S1, W1 = youla(U)
            S2, W2 = youla(V)
            # Transform the basis.
            Ptrf = np.einsum('nja, ab, nm->mjb', P1, W1, W2)
            if np.allclose(np.diag(np.diag(S2)), S2):
                # TR does not mix irreps                   
                # The structure of the irreps remains the same
                Preduced += (Ptrf, )
            else:
                # TR does mix irreps, S2 is block diagonal.
                Pnew = []
                inds = set(range(len(S2)))
                for n in range(len(S2) - 1):
                    # merge pairs of irreps that are related by TR
                    if not np.isclose(S2[n, n+1], 0):
                        Pnew.append(np.hstack([Ptrf[n], Ptrf[n+1]]))
                        inds -= {n, n+1}
                for n in inds:
                    Pnew.append(Ptrf[n])
                Preduced += (np.array(Pnew),)
        else:
            # TR mixes different irreps
            # Transform second block to make TR +-identity in the block connecting
            # P1 and P2
            blocks.remove(j)
            P2trf = np.einsum('nja, ab, nm->mjb', P2, U.T, V.T)
            Pnew = []
            for n in range(len(P1)):
                # merge pairs of irreps that are related by TR
                Pnew.append(np.hstack([P1[n], P2trf[n]]))
            Preduced += (np.array(Pnew),)

    return Preduced
