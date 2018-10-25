import numpy as np
import tinyarray as ta
import scipy.linalg as la
import itertools as it
import functools as ft
from collections import OrderedDict
import sympy
from copy import deepcopy

from .linalg import prop_to_id, _inv_int, allclose
from .model import Model


@ft.lru_cache(maxsize=100000)
def rmul(R1, R2):
    # Cached multiplication of sympy spatial parts.
    # Only called if both R1 and R2 are sympy.
    return R1 * R2


def _make_int(R):
    # If close to an integer array convert to integer tinyarray, else
    # return original array
    R_int = ta.array(np.round(R), int)
    if allclose(R, R_int):
        return R_int
    else:
        return R
    
    
def is_sympy_matrix(R):
    # Returns True if the input is a sympy.Matrix or sympy.ImmutableMatrix.
    types = [sympy.ImmutableMatrix, sympy.matrices.MatrixBase]
    if any([isinstance(R, t) for t in types]):
        return True
    else:
        return False
    

class PointGroupElement():
    """
    Class for point group elements.

    Parameters:
    -----------
    R: sympy.ImmutableMatrix or integer array
        Real space rotation action of the operator. Square matrix with size
        of the number of spatial dimensions.
    conjugate: boolean (default False)
        Whether the operation includes conplex conjugation (antiunitary operator)
    antisymmetry: boolean (default False)
        Whether the operator flips the sign of the Hamiltonian (antisymmetry)
    U: ndarray or None (default)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates
    _strict_eq : boolean, default False
        Whether to test the equality of the unitary parts when comparing with
        other PointGroupElements. By default the unitary parts are ignored.
        If True, PointGroupElements are considered equal, if the unitary parts
        are proportional, an overall phase difference is still allowed.

    Notes:
    ------
        As U is floating point and has a phase ambiguity at least,
        it is ignored when comparing objects.

        R must be provided an exact representation, either as a
        sympy.ImmutableMatrix or an integer array, as exact arithmetic is
        assumed. If R is an integer array, it must be invertible over
        the integers. Performance is much higher if integer arrays are used,
        this is always possible for crystallographic groups in the basis
        of the translation vectors.

        R is the real space rotation acion, do not include minus sign for
        the k-space action of antiunitary operators, such as time reversal.
        This minus sign will be included automatically if conjugate=True.
    """

    __slots__ = ('R', 'conjugate', 'antisymmetry', 'U', '_Rinv', '_strict_eq')

    def __init__(self, R, conjugate=False, antisymmetry=False, U=None, _strict_eq=False):
        # Make sure that R is either an immutable sympy matrix,
        # or an integer tinyarray.
        if isinstance(R, sympy.ImmutableMatrix):
            pass
        elif isinstance(R, ta.ndarray_int):
            pass
        elif isinstance(R, sympy.matrices.MatrixBase):
            R = sympy.ImmutableMatrix(R)
        elif isinstance(R, np.ndarray):
            # If it is integer, recast to integer tinyarray
            R = _make_int(R)
        else:
            raise ValueError('Real space rotation must be provided as a sympy matrix or an array.')
        self.R, self.conjugate, self.antisymmetry, self.U = R, conjugate, antisymmetry, U
        # Calculating sympy inverse is slow, remember it
        self._Rinv = None
        self._strict_eq = _strict_eq

    def __repr__(self):
        return 'PointGroupElement(\n{},\n{},\n{},\n{},\n)'.format(self.R, self.conjugate, self.antisymmetry, self.U)

    def __eq__(self, other):
        # We do not allow mixing of PointGroupElements
        # if one has a sympy spatial part R, but the other not.
        if is_sympy_matrix(self.R) ^ is_sympy_matrix(other.R):
            raise ValueError("Mixing of sympy with other types "
                             "in the spatial part R is not allowed.")
        # If Rs are of different type, convert it to numpy array
        if type(self.R) != type(other.R):
            Rs = np.array(self.R).astype(float)
            Ro = np.array(other.R).astype(float)
        else:
            Rs = self.R
            Ro = other.R
        if isinstance(Rs, np.ndarray):
            # Check equality with allclose if floating point
            R_eq = allclose(Rs, Ro)
        else:
            # If exact use exact equality
            R_eq = (Rs == Ro)
        basic_eq = R_eq and ((self.conjugate, self.antisymmetry) == (other.conjugate, other.antisymmetry))
        # Equality is only checked for U if _strict_eq is True and basic_eq is True.
        if basic_eq and (self._strict_eq is True or other._strict_eq is True):
            if (self.U is None) and (other.U is None):
                U_eq = True
            elif (self.U is None) ^ (other.U is None):
                U_eq = False
            else:
                prop, coeff = prop_to_id((self.inv() * other).U)
                U_eq = (prop and np.isclose(abs(coeff), 1))
        else:
            U_eq = True
        return basic_eq and U_eq

    def __lt__(self, other):
        # Sort group elements:
        # First by conjugate and a, then R = identity, then the rest
        # lexicographically
        Rs = ta.array(self.R, float)
        Ro = ta.array(other.R, float)
        identity = ta.array(np.eye(Rs.shape[0], dtype=int))

        if not (self.conjugate, self.antisymmetry) == (other.conjugate, other.antisymmetry):
            return (self.conjugate, self.antisymmetry) < (other.conjugate, other.antisymmetry)
        elif (Rs == identity) ^ (Ro == identity):
            return Rs == identity
        else:
            return Rs < Ro

    def __hash__(self):
        # U is not hashed, if R is floating point it is also not hashed
        R, c, a = self.R, self.conjugate, self.antisymmetry
        if isinstance(R, np.ndarray):
            return hash((c, a))
        else:
            return hash((R, c, a))

    def __mul__(self, g2):
        """Multiply two PointGroupElements"""
        g1 = self
        R1, c1, a1, U1 = g1.R, g1.conjugate, g1.antisymmetry, g1.U
        R2, c2, a2, U2 = g2.R, g2.conjugate, g2.antisymmetry, g2.U
        
        # We do not allow mixing of PointGroupElements
        # if one has a sympy spatial part R, but the other not.
        if is_sympy_matrix(R1) ^ is_sympy_matrix(R2):
            raise ValueError("Mixing of sympy with other types "
                             "in the spatial part R is not allowed.")
                
        if (U1 is None) or (U2 is None):
            U = None
        elif c1:
            U = U1.dot(U2.conj())
        else:
            U = U1.dot(U2)
        # If spatial parts are sympy matrices, use cached multiplication.
        if is_sympy_matrix(R1): # R1 and R2 are either both sympy or both not sympy.
            R = rmul(R1, R2)
        else:
            R = _make_int(np.dot(np.array(R1), np.array(R2)).astype(float))
        return PointGroupElement(R, c1^c2, a1^a2, U, _strict_eq=(self._strict_eq or g2._strict_eq))

    def __pow__(self, n):
        result = self.identity()
        g = (self if n >=0 else self.inv())
        for _ in range(abs(n)):
            result *= g
        return result

    def inv(self):
        """Invert PointGroupElement"""
        R, c, a, U = self.R, self.conjugate, self.antisymmetry, self.U
        if U is None:
            Uinv = None
        elif c:
            Uinv = la.inv(U).conj()
        else:
            Uinv = la.inv(U)
        # Check if inverse is stored, if not, calculate it
        if self._Rinv is None:
            if isinstance(R, sympy.matrices.MatrixBase):
                self._Rinv = R**(-1)
            elif isinstance(R, ta.ndarray_int):
                self._Rinv = _inv_int(R)
            elif isinstance(R, np.ndarray):
                self._Rinv = la.inv(R)
            else:  # This is probably unnecessary
                raise ValueError('Illegal datatype for the spatial part')
        result = PointGroupElement(self._Rinv, c, a, Uinv, _strict_eq=self._strict_eq)
        result._Rinv = R
        return result

    def _strictereq(self, other):
        # Stricter equality, testing the unitary parts to be approx. equal
        if (self.U is None) or (other.U is None):
            return False
        return ((self.R, self.conjugate, self.antisymmetry) == (other.R, other.conjugate, other.antisymmetry) and
                allclose(self.U, other.U))

    def apply(self, model):
        """Return copy of model with applied symmetry operation.

        if unitary: (+/-) U H(inv(R) k) U^dagger
        if antiunitary: (+/-) U H(- inv(R) k).conj() U^dagger

        (+/-) stands for (symmetry / antisymmetry)

        If self.U is None, U is taken as the identity.
        """
        R, antiunitary, antisymmetry, U = self.R, self.conjugate, self.antisymmetry, self.U
        if isinstance(R, sympy.matrices.MatrixBase):
            R = R**(-1)
        elif isinstance(R, ta.ndarray_int):
            R = _inv_int(R)
        else:
            R = la.inv(R)
        R *= (-1 if antiunitary else 1)
        result = model.rotate_momenta(R)
        if antiunitary:
            result = result.conj()
        if antisymmetry:
            result = -result
        if U is not None:
            result = U * result * U.T.conj()

        return result

    def identity(self):
        """Return identity element with the same structure as self."""
        dim = self.R.shape[0]
        if isinstance(self.R, sympy.matrices.MatrixBase):
            R = sympy.ImmutableMatrix(sympy.eye(dim))
        elif isinstance(self.R, ta.ndarray_int):
            R = ta.identity(dim, int)
        else:
            R = np.eye(dim, dtype=float)
        if self.U is not None:
            U = np.eye(self.U.shape[0])
        else:
            U = None
        return PointGroupElement(R, False, False, U)


def identity(dim, shape=None):
    """Return identity operator with appropriate shape.

    Parameters:
    -----------
    dim : int
        Dimension of real space.
    shape : int or None
        Size of the unitary part of the operator. If None,
        U = None."""
    R = ta.identity(dim, int)
    if shape is not None:
        U = np.eye(shape)
    else:
        U = None
    return PointGroupElement(R, False, False, U)


class ContinuousGroupGenerator():
    """
    Class for continuous group generators.
    Generates family of symmetry operators that act on
    the Hamiltonian as
    H(k) -> exp(-1j x U) H(exp(1j x R) k) exp(1j x U)
    with x real parameter.

    Parameters:
    -----------
    R: ndarray or None (default)
        Real space rotation generator, Hermitian antisymmetric.
        None corresponds to the zero matrix.
    U: ndarray or None (default)
        Hilbert space unitary rotation generator, Hermitian.
        None corresponds to the zero matrix.
    """

    __slots__ = ('R', 'U')

    def __init__(self, R=None, U=None):
        # Make sure that R and U have correct properties
        if not ((R is None or (allclose(R, -R.T) and allclose(R, R.T.conj())))
                and (U is None or allclose(U, U.T.conj()))):
            raise ValueError('R must be Hermitian antisymmetric and U Hermitian or None.')
        self.R, self.U = R, U

    def __repr__(self):
        return 'ContinuousGroupGenerator(\n{},\n{},\n)'.format(self.R, self.U)

    def apply(self, monomials):
        """Return copy of monomials with applied infinitesimal generator.
        1j * (H(k) U - U H(k)) + 1j * dH(k)/dk_i R_{ij} k_j
        """
        R, U = self.R, self.U
        momenta = monomials.momenta
        R_nonzero = not (R is None or allclose(R, 0))
        U_nonzero = not (U is None or allclose(U, 0))
        if R_nonzero:
            dim = R.shape[0]
            assert len(momenta) == dim
        result = monomials.zeros_like()
        if R_nonzero:
            def trf(key):
                return sum([sympy.diff(key, momenta[i]) * R[i, j] * momenta[j]
                          for i, j in it.product(range(dim), repeat=2)])
            result += 1j * monomials.transform_symbolic(trf)
        if U_nonzero:
            result += monomials * (1j*U) + (-1j*U) * monomials
        return result


## General group theory algorithms

def generate_group(gens):
    """Generate group from gens

    Parameters:
    -----------
    gens : iterable of PointGroupElement
        generator set of the group

    Returns:
    --------
    group : set of PointGroupElement
        group generated by gens, closed under multiplication
    """
    gens = set(gens.copy())
    # here we keep all the elements generated so far
    group = gens.copy()
    # these are the elements generated in the previous step
    oldgroup = gens.copy()
    while True:
        # generate new elements by multiplying old elements with generators
        newgroup = {a * b for a, b in it.product(oldgroup, gens)}
        # only keep those that are new
        newgroup -= group
        # if there are any new, add them to group and set them as old
        if len(newgroup) > 0:
            group |= newgroup
            oldgroup = newgroup.copy()
        # if there were no new elements, we are done
        else:
            break
    return set(group)


def set_multiply(G, H):
    # multiply sets of group elements
    return {g * h for g, h in it.product(G, H)}


def generate_subgroups(group):
    """Generate all subgroups of group, including the trivial group
    and itself.

    Parameters:
    -----------
    group : set of PointGroupElement
        A closed group as set of its elements.

    Returns:
    --------
    subgroups : dict of forzenset: set
        frozesets are the subgroups, sets are a generator set of the
        subgroup.
    """
    # Make all the cyclic subgroups generated by one element
    sg1 = {frozenset(generate_group({g})): {g} for g in group}
    # Iteratively generate all subgroups generated by more elements
    # here we keep all the subgroups generated so far
    subgroups = sg1.copy()
    # these are the subgroups generated in the previous step
    sgold = sg1.copy()

    while True:
        sgnew = dict()
        # extend subgroups from previous step by all cyclic groups
        for (sg, gen), (g1, gen1) in it.product(sgold.items(), sg1.items()):
            # if cyclic group is already contained in group,
            # no point extending by it
            if not g1 <= sg:
                newsg = frozenset(generate_group(gen | gen1))
                # don't do anything if it is already generated
                # with less or equal number of generators
                if newsg not in subgroups:
                    # If we managed to extend anything, we append
                    # it to the list of subgroups and new subgroups
                    newgen = gen | gen1
                    subgroups[newsg] = newgen
                    sgnew[newsg] = newgen
        if len(sgnew) > 0:
            sgold = sgnew.copy()
        # If no extension, or we are already at the full group, stop
        if (len(sgnew) == 0 or
            min([len(sg) for sg in sgnew.keys()]) == len(group)):
            break

    return subgroups


## Predefined point groups

def cubic(tr=True, ph=True, generators=False):
    """Generate cubic point group in standard basis.
    Parameters:
    -----------
    tr, ph : bool (default True)
        Whether to include time-reversal and particle-hole
        symmetry.
    generators : bool (default false)
        Only return the group generators if True.

    Returns:
    --------
    set of PointGroupElement objects with integer rotations"""
    eye = ta.array(np.eye(3), int)
    E = PointGroupElement(eye, False, False, None)
    I = PointGroupElement(-eye, False, False, None)
    C4 = PointGroupElement(ta.array([[1, 0, 0],
                                     [0, 0, 1],
                                     [0, -1, 0]], int), False, False, None)
    C3 = PointGroupElement(ta.array([[0, 0, 1],
                                     [1, 0, 0],
                                     [0, 1, 0]], int), False, False, None)
    cubic_gens = {I, C4, C3}
    if tr:
        TR = PointGroupElement(eye, True, False, None)
        cubic_gens.add(TR)
    if ph:
        PH = PointGroupElement(eye, True, True, None)
        cubic_gens.add(PH)
    if generators:
        return cubic_gens
    else:
        return generate_group(cubic_gens)


def hexagonal(tr=True, ph=True, generators=False, sympy_R=True):
    """Generate hexagonal point group in standard basis.
    Parameters:
    -----------
    tr, ph : bool (default True)
        Whether to include time-reversal and particle-hole
        symmetry.
    generators : bool (default True)
        Only return the group generators if True.
    sympy_R: bool (default True)
        Whether the rotation matrices should be exact sympy
        representations.

    Returns:
    --------
    set of PointGroupElement objects with Sympy rotations."""

    if sympy_R:
        eye = sympy.ImmutableMatrix(sympy.eye(2))
        Mx = PointGroupElement(sympy.ImmutableMatrix([[-1, 0],
                                                      [0, 1]]),
                                   False, False, None)
        C6 = PointGroupElement(sympy.ImmutableMatrix(
                                    [[sympy.Rational(1, 2), sympy.sqrt(3)/2],
                                     [-sympy.sqrt(3)/2,       sympy.Rational(1, 2)]]
                                                     ),
                                     False, False, None)
    else:
        eye = np.eye(2)
        Mx = PointGroupElement(np.diag([-1, 1]), False, False, None)
        C6 = PointGroupElement(np.array(
                                    [[1/2, np.sqrt(3)/2],
                                     [-np.sqrt(3)/2, 1/2]]
                                                     ),
                                     False, False, None)
    gens_hex_2D ={Mx, C6}
    if tr:
        TR = PointGroupElement(eye, True, False, None)
        gens_hex_2D.add(TR)
    if ph:
        PH = PointGroupElement(eye, True, True, None)
        gens_hex_2D.add(PH)
    if generators:
        return gens_hex_2D
    else:
        return generate_group(gens_hex_2D)


## Human readable group element names

def name_PG_elements(g):
    """Automatically name PG operators in 3D cubic group"""

    def find_order(g):
        dim = g.shape[0]
        gpower = g
        order = 1
        while not allclose(gpower, np.eye(dim)):
            gpower = g.dot(gpower)
            order += 1
        return order

    def find_axis(g):
        eig = la.eig(g)
        n = [n for i, n in enumerate(eig[1].T) if np.isclose(eig[0][i], 1)]
        assert len(n) == 1
        n = n[0]
        if not np.isclose(n[0], 0):
            n = np.real(n/n[0])
        elif not np.isclose(n[1], 0):
            n = np.real(n/n[1])
        else:
            n = np.real(n/n[2])
        return n

    def axisname(n):
        epsilon = 0.2
        return ''.join([(s if n[i] > epsilon else '') +
                    ('-' + s if n[i] < -epsilon else '')
                    for i, s in enumerate(['x', 'y', 'z'])])

    def find_direction(g, n):
        if allclose(g.dot(g), np.eye(g.shape[0])):
            return ''
        v = np.random.random(3)
        p = np.cross(v, g.dot(v)).dot(n)
        if p < 0.0001:
            return '-'
        else:
            return ''

    name = ''
    if g.conjugate and not g.antisymmetry:
        name += 'T'
    if g.conjugate and g.antisymmetry:
        name += 'P'

    if not g.conjugate and g.antisymmetry:
        name += 'TP'
    g = np.array(g.R).astype(np.float_)
    # if less than 3D, pad with identity
    if g.shape[0] < 3:
        g = la.block_diag(g, np.eye(3 - g.shape[0]))
    dim = g.shape[0]
    det = la.det(g)
    evals = la.eigvals(g)
    if np.isclose(det, 1):
        if allclose(g, np.eye(dim)):
            name += 'Id'
        else:
            n = find_axis(g)
            name += 'C' + find_direction(g, n) + str(find_order(g)) + axisname(n)
    else:
        if allclose(g, -np.eye(dim)):
            name += 'I'
        else:
            order = find_order(-g)
            n = find_axis(-g)
            if order == 2:
                name += 'M' + axisname(n)
            else:
                name += 'S' + find_direction(g, n) + str(order) + axisname(n)
    return name


## Predefined representations

def spin_matrices(s, include_0=False):
    """Construct spin-s matrices for any half-integer spin.
    If include_0 is True, S[0] is the identity, indices 1, 2, 3
    correspond to x, y, z. Otherwise indices 0, 1, 2 are x, y, z.
    """
    d = np.round(2*s + 1)
    assert np.isclose(d, int(d))
    d = int(d)
    Sz = 1/2 * np.diag(np.arange(d - 1, -d, -2))
    # first diagonal for general s from en.wikipedia.org/wiki/Spin_(physics)
    diag = [1/2*np.sqrt((s + 1) * 2*i - i * (i + 1)) for i in np.arange(1, d)]
    Sx = np.diag(diag, k=1) + np.diag(diag, k=-1)
    Sy = -1j*np.diag(diag, k=1) + 1j*np.diag(diag, k=-1)
    if include_0:
        return np.array([np.eye(d), Sx, Sy, Sz])
    else:
        return np.array([Sx, Sy, Sz])


def spin_rotation(n, S, roundint=False):
    """Construct the unitary spin rotation matrix for rotation generators S
    with rotation specified by the vector n. It is assumed that
    n and S have the same first dimension.
    If roundint is True, result is converted to integer tinyarray if possible.
    """
    # Make matrix exponential for rotation representation
    U = la.expm(-1j * np.tensordot(n, S , axes=((0), (0))))
    if roundint:
        Ur = np.round(np.real(U))
        assert allclose(U, Ur)
        U = ta.array(Ur.astype(int))
    return U


def L_matrices(d=3, l=1):
    """Construct real space rotation generator matrices in d=2 or 3 dimensions.
    Can also be used to get angular momentum operators for real atomic orbitals
    in 3 dimensions, for p-orbitals use `l=1`, for d-orbitals `l=2`. The basis
    of p-orbitals is `p_x`, `p_y`, `p_z`, for d-orbitals `d_{x^2 - y^2}`,
    `d_{3 z^2 - r^2}`, `d_{xy}`, `d_{yz}`, `d_{zx}`. The matrices are all
    purely imaginary and antisymmetric.
    To generate finite rotations, use 'spin_rotation(n, L)'.
    """
    if d == 2 and l==1:
        return 1j * np.array([[[0, -1],
                              [1, 0]]])
    elif d == 3 and l==1:
        return 1j * np.array([[[0, 0, 0],
                               [0, 0, -1],
                               [0, 1, 0]],
                              [[0, 0, 1],
                               [0, 0, 0],
                               [-1, 0, 0]],
                              [[0, -1, 0],
                               [1, 0, 0],
                               [0, 0, 0]]])
    elif d == 3 and l==2:
        s3 = np.sqrt(3)
        return 1j * np.array([[[0, 0, 0, -1, 0],
                              [0, 0, 0, -s3, 0],
                              [0, 0, 0, 0, 1],
                              [1, s3, 0, 0, 0],
                              [0, 0, -1, 0, 0]],
                             [[0, 0, 0, 0, -1],
                              [0, 0, 0, 0, s3],
                              [0, 0, 0, -1, 0],
                              [0, 0, 1, 0, 0],
                              [1, -s3, 0, 0, 0]],
                             [[0, 0, -2, 0, 0],
                              [0, 0, 0, 0, 0],
                              [2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, -1, 0]]])
    else:
        raise ValueError('Only 2 and 3 dimensions are supported.')

## Utility function for lattice models

def symmetry_from_permutation(R, perm, norbs, onsite_action=None,
                              antiunitary=False, antisymmetry=False):
    """Construct symmetry operator for lattice systems with multiple sites.

    Parameters:
    -----------
    R : real space rotation
    perm : dict : {site: image_site}
        permutation of the sites under the symmetry action
    norbs : OrderedDict : {site : norbs_site} or tuple of tuples ((site, norbs_site), )
        sites are ordered in the order specified, with blocks of size norbs_site
        corresponding to each site.
    onsite_action : dict : {site: ndarray} or ndarray or None
        onsite symmetry action, such as spin rotation for each site. If only one
        array is specified, it is used for every site. If None (default), identity
        is used on every site. Size of the arrays must be consistent with norbs.
    antiunitary, antisymmetry : bool

    Returns:
    --------
    g : PointGroupElement
        PointGroupElement corresponding to the operation.

    Notes:
    ------
    Sites can be indexed by any hashable identifiers, such as integers or stings.
    """
    norbs = OrderedDict(norbs)
    ranges = dict()
    N = 0
    for a, n in norbs.items():
        ranges[a] = slice(N, N + n)
        N += n

    if onsite_action is None:
        onsite_action = {a: np.eye(n) for a, n in norbs.items()}
    elif isinstance(onsite_action, np.ndarray):
        onsite_action = {a: onsite_action for a in norbs.keys()}

    # Build transformation matrix
    if not set(perm.keys()) == set(perm.values()) == set(norbs.keys()):
        raise ValueError('perm keys, values and norbs keys must contain the same sites.')
    U = np.zeros((N, N), dtype=complex)
    for a in norbs.keys():
        if not norbs[a] == norbs[perm[a]]:
            raise ValueError('Symmetry related sites must have the same number or orbitals')
        U[ranges[a], ranges[perm[a]]] = onsite_action[a]
    g = PointGroupElement(R, antiunitary, antisymmetry, U)
    return g
