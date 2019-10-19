# -*- coding: utf-8 -*-

import numpy as np
import tinyarray as ta
import scipy.linalg as la
from itertools import product
from functools import lru_cache
from fractions import Fraction
from numbers import Number
from collections import OrderedDict
import sympy
from copy import deepcopy

from .linalg import prop_to_id, _inv_int, allclose
from .model import Model


# Cache the operations that are potentially slow and happen a lot in
# group theory applications. Essentially store the multiplication table.
@lru_cache(maxsize=10000)
def _mul(R1, R2):
    # Cached multiplication of spatial parts.
    if is_sympy_matrix(R1) and is_sympy_matrix(R2):
        # If spatial parts are sympy matrices, use cached multiplication.
        R = R1 * R2
    elif not (is_sympy_matrix(R1) or is_sympy_matrix(R2)):
        # If arrays, use dot
        R = ta.dot(R1, R2)
    elif ((is_sympy_matrix(R1) or is_sympy_matrix(R2)) and
        (isinstance(R1, ta.ndarray_int) or isinstance(R2, ta.ndarray_int))):
        # Multiplying sympy and integer tinyarray is ok, should result in sympy
        R = sympy.ImmutableMatrix(R1) * sympy.ImmutableMatrix(R2)
    else:
        raise ValueError("Mixing of sympy and floating point in the spatial part R is not allowed. "
                         "To avoid this error, make sure that all PointGroupElements are initialized "
                         "with either floating point arrays or sympy matrices as rotations. "
                         "Integer arrays are allowed in both cases.")
    R = _make_int(R)
    return R


@lru_cache(maxsize=1000)
def _inv(R):
    if isinstance(R, ta.ndarray_int):
        Rinv = ta.array(_inv_int(R), int)
    elif isinstance(R, ta.ndarray_float):
        Rinv = ta.array(la.inv(R))
    elif is_sympy_matrix(R):
        Rinv = R**(-1)
    else:
        raise ValueError('Invalid rotation matrix.')
    return Rinv


@lru_cache(maxsize=10000)
def _eq(R1, R2):
    if type(R1) != type(R2):
        R1 = ta.array(np.array(R1).astype(float), float)
        R2 = ta.array(np.array(R2).astype(float), float)
    if isinstance(R1, ta.ndarray_float):
        # Check equality with allclose if floating point
        R_eq = allclose(R1, R2)
    else:
        # If exact use exact equality
        R_eq = (R1 == R2)
    return R_eq


@lru_cache(maxsize=1000)
def _make_int(R):
    # If close to an integer array convert to integer tinyarray, else
    # return original array
    R_float = (np.array(R).astype(float) if is_sympy_matrix(R) else R)
    R_int = ta.array(np.round(R_float), int)
    if allclose(R_float, R_int):
        return R_int
    else:
        return R


def _is_hermitian(a):
    return allclose(a, a.conjugate().transpose())


def _is_antisymmetric(a):
    return allclose(a, -a.transpose())


def is_sympy_matrix(R):
    # Returns True if the input is a sympy.Matrix or sympy.ImmutableMatrix.
    return isinstance(R, (sympy.ImmutableMatrix, sympy.matrices.MatrixBase))


class PointGroupElement:
    """An element of a point group.

    Parameters
    ----------
    R : sympy.ImmutableMatrix or array
        Real space rotation action of the operator. Square matrix with size
        of the number of spatial dimensions.
    conjugate : boolean (default False)
        Whether the operation includes conplex conjugation (antiunitary operator)
    antisymmetry : boolean (default False)
        Whether the operator flips the sign of the Hamiltonian (antisymmetry)
    U : ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates
    _strict_eq : boolean (default False)
        Whether to test the equality of the unitary parts when comparing with
        other PointGroupElements. By default the unitary parts are ignored.
        If True, PointGroupElements are considered equal, if the unitary parts
        are proportional, an overall phase difference is still allowed.

    Notes
    -----
    As U is floating point and has a phase ambiguity at least, hence
    it is ignored when comparing objects by default.

    R is the real space rotation acion. Do not include minus sign for
    the k-space action of antiunitary operators, such as time reversal.
    This minus sign will be included automatically if 'conjugate=True'.

    For most uses R can be provided as a floating point array. It is
    necessary to use exact sympy matrix representation if the PGE has
    to act on Models with complicated momentum dependence (not polynomial),
    as the function parts of models are compared exactly. If the momentum
    dependence is periodic (sine, cosine and exponential), use BlochModel,
    this works with floating point rotations.
    """

    __slots__ = ('R', 'conjugate', 'antisymmetry', 'U', '_strict_eq')

    def __init__(self, R, conjugate=False, antisymmetry=False, U=None, _strict_eq=False):
        if isinstance(R, sympy.ImmutableMatrix):
            # If it is integer, recast to integer tinyarray
            R = _make_int(R)
        elif isinstance(R, ta.ndarray_int):
            pass
        elif isinstance(R, ta.ndarray_float):
            R = _make_int(R)
        elif isinstance(R, sympy.matrices.MatrixBase):
            R = sympy.ImmutableMatrix(R)
            R = _make_int(R)
        elif isinstance(R, np.ndarray):
            # If it is integer, recast to integer tinyarray
            R = ta.array(R)
            R = _make_int(R)
        else:
            raise ValueError('Real space rotation must be provided as a sympy matrix or an array.')
        self.R, self.conjugate, self.antisymmetry, self.U = R, conjugate, antisymmetry, U
        # Calculating sympy inverse is slow, remember it
        self._strict_eq = _strict_eq

    def __repr__(self):
        return ('\nPointGroupElement(\nR = {},\nconjugate = {},\nantisymmetry = {},\nU = {})'
                .format(repr(self.R).replace('\n', '\n    '),
                        self.conjugate,
                        self.antisymmetry,
                        repr(self.U).replace('\n', '\n    ') if self.U is not None else 'None'))

    def __str__(self):
        return pretty_print_pge(self, full=True)

    def _repr_latex_(self):
        return pretty_print_pge(self, full=False, latex=True)

    def _repr_pretty_(self, pp, cycle):
        pp.text(pretty_print_pge(self, full=False))

    def __eq__(self, other):
        R_eq = _eq(self.R, other.R)
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
        Rs = ta.array(np.array(self.R).astype(float), float)
        Ro = ta.array(np.array(other.R).astype(float), float)
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
        if isinstance(R, ta.ndarray_float):
            return hash((c, a))
        else:
            return hash((R, c, a))

    def __mul__(self, g2):
        g1 = self
        R1, c1, a1, U1 = g1.R, g1.conjugate, g1.antisymmetry, g1.U
        R2, c2, a2, U2 = g2.R, g2.conjugate, g2.antisymmetry, g2.U

        if (U1 is None) or (U2 is None):
            U = None
        elif c1:
            U = U1.dot(U2.conj())
        else:
            U = U1.dot(U2)
        R = _mul(R1, R2)
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
        Rinv = _inv(R)
        result = PointGroupElement(Rinv, c, a, Uinv, _strict_eq=self._strict_eq)
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
        R = _inv(R)
        R = R * (-1 if antiunitary else 1)
        result = model.rotate_momenta(R)
        if antiunitary:
            result = result.conj()
        if antisymmetry:
            result = -result
        if U is not None:
            result = U @ result @ U.T.conj()

        return result

    def identity(self):
        """Return identity element with the same structure as self."""
        dim = self.R.shape[0]
        R = ta.identity(dim, int)
        if self.U is not None:
            U = np.eye(self.U.shape[0])
        else:
            U = None
        return PointGroupElement(R, False, False, U)

## Factory functions for point group elements

def identity(dim, shape=None):
    """Return identity operator with appropriate shape.

    Parameters
    ----------
    dim : int
        Dimension of real space.
    shape : int (optional)
        Size of the unitary part of the operator.
        If not provided, U is set to None.

    Returns
    -------
    id : PointGroupElement
    """
    R = ta.identity(dim, int)
    if shape is not None:
        U = np.eye(shape)
    else:
        U = None
    return PointGroupElement(R, False, False, U)


def time_reversal(realspace_dim, U=None, spin=None):
    """Return a time-reversal symmetry operator

    parameters
    ----------
    realspace_dim : int
        Realspace dimension
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates.
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the time reversal
        operator. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of time-reversal operator is `U = exp(-i π s_y)`. Only one of `U` and `spin`
        may be provided.

    Returns
    -------
    T : PointGroupElement
    """
    if U is not None and spin is not None:
        raise ValueError('Only one of `U` and `spin` may be provided.')
    if spin is not None:
        U = spin_rotation(np.pi * np.array([0, 1, 0]), spin)
    R = ta.identity(realspace_dim, int)
    return PointGroupElement(R, conjugate=True, antisymmetry=False, U=U)


def particle_hole(realspace_dim, U=None):
    """Return a particle-hole symmetry operator

    parameters
    ----------
    realspace_dim : int
        Realspace dimension
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates

    Returns
    -------
    P : PointGroupElement
    """
    R = ta.identity(realspace_dim, int)
    return PointGroupElement(R, conjugate=True, antisymmetry=True, U=U)


def chiral(realspace_dim, U=None):
    """Return a chiral symmetry operator

    parameters
    ----------
    realspace_dim : int
        Realspace dimension
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates

    Returns
    -------
    P : PointGroupElement
    """
    R = ta.identity(realspace_dim, int)
    return PointGroupElement(R, conjugate=False, antisymmetry=True, U=U)


def inversion(realspace_dim, U=None):
    """Return an inversion operator

    parameters
    ----------
    realspace_dim : int
        Realspace dimension
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates

    Returns
    -------
    P : PointGroupElement
    """
    R = -ta.identity(realspace_dim, int)
    return PointGroupElement(R, conjugate=False, antisymmetry=False, U=U)


def rotation(angle, axis=None, inversion=False, U=None, spin=None):
    """Return a rotation operator

    parameters
    ----------
    angle : float
        Rotation angle in units of 2 pi.
    axis : ndarray or None (default)
        Rotation axis, optional. If not provided, a 2D rotation is generated
        around the axis normal to the plane. If a 3D vector is provided,
        a 3D rotation is generated around this axis. Does not need to be
        normalized to 1.
    inversion : bool (default False)
        Whether to generate a rotoinversion. By default a proper rotation
        is returned. Only valid in 3D.
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the
        operator. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of rotation operator is `U = exp(-i n⋅s)`. In 2D the z axis is assumed to be
        the rotation axis. Only one of `U` and `spin` may be provided.

    Returns
    -------
    P : PointGroupElement
    """
    if U is not None and spin is not None:
        raise ValueError('Only one of `U` and `spin` may be provided.')

    angle = 2 * np.pi * angle
    if axis is None:
        # 2D
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        if spin is not None:
            U = spin_rotation(angle * np.array([0, 0, 1]), spin)
    elif len(axis) == 3:
        # 3D
        n = angle * np.array(axis, float) / la.norm(axis)
        R = spin_rotation(n, L_matrices(d=3, l=1))
        R *= (-1 if inversion else 1)
        if spin is not None:
            U = spin_rotation(n, spin)
    else:
        raise ValueError('`axis` needs to be `None` or a 3D vector.')
    return PointGroupElement(R.real, conjugate=False, antisymmetry=False, U=U)


def mirror(axis, U=None, spin=None):
    """Return a mirror operator

    Parameters
    ----------
    axis : ndarray
        Normal of the mirror. The dimensionality of the operator is the same
        as the length of `axis`.
    U: ndarray (optional)
        The unitary action on the Hilbert space.
        May be None, to be able to treat symmetry candidates
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the
        operator. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of mirror operator is `U = exp(-i π n⋅s)` where n is normalized to 1. In 2D the
        axis is treated as x and y coordinates. Only one of `U` and `spin` may be provided.

    Returns
    -------
    P : PointGroupElement

    Notes:
    ------
        Warning: in 2D the real space action of a mirror and and a 2-fold rotation
        around an axis in the plane is identical, however the action on angular momentum
        is different. Here we consider the action of the mirror, which is the same as the
        action of a 2-fold rotation around the mirror axis.
    """
    if U is not None and spin is not None:
        raise ValueError('Only one of `U` and `spin` may be provided.')

    axis = np.array(axis, float)
    axis /= la.norm(axis)
    R = np.eye(axis.shape[0]) - 2 * np.outer(axis, axis)
    if spin is not None:
        if len(axis) == 2:
            axis = np.append(axis, 0)
        U = spin_rotation(np.pi * axis, spin)

    return PointGroupElement(R, conjugate=False, antisymmetry=False, U=U)

## Continuous symmetry generators (conserved quantities)

class ContinuousGroupGenerator:
    r"""A generator of a continuous group.

    Generates a family of symmetry operators that act on the Hamiltonian as:

    .. math:: H(k) → \exp{-iλU} H(\exp{iλR} k) \exp{iλU}

    with λ a real parameter.

    Parameters
    ----------
    R: ndarray (optional)
        Real space rotation generator: Hermitian antisymmetric.
        If not provided, the zero matrix is used.
    U: ndarray (optional)
        Hilbert space unitary rotation generator: Hermitian.
        If not provided, the zero matrix is used.
    """

    __slots__ = ('R', 'U')

    def __init__(self, R=None, U=None):
        # Make sure that R and U have correct properties
        if R is not None and not _is_hermitian(R) and not _is_antisymmetric(R):
            raise ValueError('R must be Hermitian antisymmetric')
        if U is not None and not _is_hermitian(U):
            raise ValueError('U must be Hermitian')
        self.R, self.U = R, U

    def __repr__(self):
        return ('\nContinuousGroupGenerator(\nR = {},\nU = {})'
                .format(repr(self.R).replace('\n', '\n    ') if self.R is not None else 'None',
                        repr(self.U).replace('\n', '\n    ') if self.U is not None else 'None'))

    def __str__(self):
        return pretty_print_cgg(self, latex=False)

    def _repr_latex_(self):
        return pretty_print_cgg(self, latex=True)

    def _repr_pretty_(self, pp, cycle):
        pp.text(pretty_print_cgg(self))

    def apply(self, model):
        """Return copy of model `H(k)` with applied infinitesimal generator.
        1j * (H(k) U - U H(k)) + 1j * dH(k)/dk_i R_{ij} k_j
        """
        R, U = self.R, self.U
        momenta = model.momenta
        R_nonzero = not (R is None or allclose(R, 0))
        U_nonzero = not (U is None or allclose(U, 0))
        if R_nonzero:
            dim = R.shape[0]
            assert len(momenta) == dim
        result = model.zeros_like()
        if R_nonzero:
            def trf(key):
                return sum([sympy.diff(key, momenta[i]) * R[i, j] * momenta[j]
                          for i, j in product(range(dim), repeat=2)])
            result += 1j * model.transform_symbolic(trf)
        if U_nonzero:
            result += model @ (1j*U) + (-1j*U) @ model
        return result


## General group theory algorithms

def generate_group(gens):
    """Generate group from gens

    Parameters
    ----------
    gens : iterable of PointGroupElement
        generator set of the group

    Returns
    -------
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
        newgroup = {a * b for a, b in product(oldgroup, gens)}
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
    return {g * h for g, h in product(G, H)}


def generate_subgroups(group):
    """Generate all subgroups of group, including the trivial group
    and itself.

    Parameters
    ----------
    group : set of PointGroupElement
        A closed group as set of its elements.

    Returns
    -------
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
        for (sg, gen), (g1, gen1) in product(sgold.items(), sg1.items()):
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

def square(tr=True, ph=True, generators=False, spin=None):
    """
    Generate square point group in standard basis.

    Parameters
    ----------
    tr, ph : bool (default True)
        Whether to include time-reversal and particle-hole
        symmetry.
    generators : bool (default false)
        Only return the group generators if True.
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the
        operator. If not provided, the PointGroupElements have the unitary
        action set to None. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of rotation operator is `U = exp(-i n⋅s)`. In 2D the z axis is assumed to be
        the rotation axis. If `ph` is True, `spin` may not be provided, as it is not
        possible to deduce the unitary representation of particle-hole symmetry from
        spin alone. In this case construct the particle-hole operator manually.

    Returns
    -------
    set of PointGroupElement objects with integer rotations

    Notes:
    ------
        Warning: in 2D the real space action of a mirror and and a 2-fold rotation
        around an axis in the plane is identical, however the action on angular  momentum
        is different. Here we consider the action of the mirror, which is the same as the
        action of a 2-fold rotation around the mirror axis, assuming inversion acts trivially.
    """
    if ph and spin is not None:
        raise ValueError('If `ph` is True, `spin` may not be provided, as it is not '
                         'possible to deduce the unitary representation of particle-hole symmetry '
                         'from spin alone. In this case construct the particle-hole operator manually.')
    Mx = mirror([1, 0], spin=spin)
    C4 = rotation(1/4, spin=spin)
    gens = {Mx, C4}
    if tr:
        TR = time_reversal(2, spin=spin)
        gens.add(TR)
    if ph:
        PH = particle_hole(2)
        gens.add(PH)
    if generators:
        return gens
    else:
        return generate_group(gens)


def cubic(tr=True, ph=True, generators=False, spin=None):
    """
    Generate cubic point group in standard basis.

    Parameters
    ----------
    tr, ph : bool (default True)
        Whether to include time-reversal and particle-hole
        symmetry.
    generators : bool (default false)
        Only return the group generators if True.
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the
        operator. If not provided, the PointGroupElements have the unitary
        action set to None. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of rotation operator is `U = exp(-i n⋅s)`. If `ph` is True, `spin` may not be
        provided, as it is not  possible to deduce the unitary representation of
        particle-hole symmetry from spin alone. In this case construct the
        particle-hole operator manually.

    Returns
    -------
    set of PointGroupElement objects with integer rotations

    Notes:
    ------
        We assume inversion acts trivially in spin space.
    """
    if ph and spin is not None:
        raise ValueError('If `ph` is True, `spin` may not be provided, as it is not '
                         'possible to deduce the unitary representation of particle-hole symmetry '
                         'from spin alone. In this case construct the particle-hole operator manually.')
    I = inversion(3, U=(None if spin is None else spin_rotation(np.zeros(3), spin)))
    C4 = rotation(1/4, [1, 0, 0], spin=spin)
    C3 = rotation(1/3, [1, 1, 1], spin=spin)
    cubic_gens = {I, C4, C3}
    if tr:
        TR = time_reversal(3, spin=spin)
        cubic_gens.add(TR)
    if ph:
        PH = particle_hole(3)
        cubic_gens.add(PH)
    if generators:
        return cubic_gens
    else:
        return generate_group(cubic_gens)


def hexagonal(dim=2, tr=True, ph=True, generators=False, sympy_R=True, spin=None):
    """
    Generate hexagonal point group in standard basis in 2 or 3 dimensions.
    Mirror symmetries with the main coordinate axes as normals are included.
    In 3D the hexagonal axis is the z axis.

    Parameters
    ----------
    dim : int (default 2)
        Real sapce dimensionality, 2 or 3.
    tr, ph : bool (default True)
        Whether to include time-reversal and particle-hole
        symmetry.
    generators : bool (default True)
        Only return the group generators if True.
    sympy_R: bool (default True)
        Whether the rotation matrices should be exact sympy
        representations.
    spin : float or sequence of arrays (optional)
        Spin representation to use for the unitary action of the
        operator. If not provided, the PointGroupElements have the unitary
        action set to None. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator. The unitary action
        of rotation operator is `U = exp(-i n⋅s)`. In 2D the z axis is assumed to be
        the rotation axis. If `ph` is True, `spin` may not be provided, as it is not
        possible to deduce the unitary representation of particle-hole symmetry from
        spin alone. In this case construct the particle-hole operator manually.

    Returns
    -------
    set of PointGroupElements

    Notes:
    ------
        Warning: in 2D the real space action of a mirror and and a 2-fold rotation
        around an axis in the plane is identical, however the action on angular momentum
        is different. Here we consider the action of the mirror, which is the same as the
        action of a 2-fold rotation around the mirror axis, assuming inversion acts trivially.
    """
    if spin is not None and sympy_R:
        U6 = spin_rotation(np.pi / 3 * np.array([0, 0, 1]), spin)
    else:
        U6 = None
    if dim == 2:
        Mx = mirror([1, 0], spin=spin)
        if sympy_R:

            C6 = PointGroupElement(sympy.ImmutableMatrix(
                                        [[sympy.Rational(1, 2), sympy.sqrt(3)/2],
                                         [-sympy.sqrt(3)/2,       sympy.Rational(1, 2)]]
                                                         ),
                                         False, False, U6)
        else:
            C6 = rotation(1/6, spin=spin)
        gens = {Mx, C6}
    elif dim == 3:
        I = inversion(3, U=(None if spin is None else spin_rotation(np.zeros(3), spin)))
        C2x = rotation(1/2, [1, 0, 0], spin=spin)
        if sympy_R:
            C6 = PointGroupElement(sympy.ImmutableMatrix(
                                        [[sympy.Rational(1, 2), sympy.sqrt(3)/2, 0],
                                         [-sympy.sqrt(3)/2, sympy.Rational(1, 2), 0],
                                         [0, 0, 1]]
                                                         ),
                                         False, False, U6)
        else:
            C6 = rotation(1/6, [0, 0, 1], spin=spin)
        gens = {I, C2x, C6}
    else:
        raise ValueError('Only 2 and 3 dimensions are supported.')

    if tr:
        TR = time_reversal(dim, spin=spin)
        gens.add(TR)
    if ph:
        PH = particle_hole(dim)
        gens.add(PH)
    if generators:
        return gens
    else:
        return generate_group(gens)


## Human readable group element names

def pretty_print_pge(g, full=False, latex=False):
    """
    Return a human readable string representation of PointGroupElement

    Parameters
    ----------

    g : PointGroupElement
        Point group element to be represented.
    full : bool (default False)
        Whether to return a full representation.
        The default short representation only contains the real space action
        and the symbol of the Altland-Zirnbauer part of the symmetry (see below).
        The full representation presents the symmetry action on the Hamiltonian
        and the unitary Hilbert-space action if set.
    latex : bool (default False)
        Whether to output LateX formatted string.

    Returns
    -------
    name : string
        In the short representation it is a sting `rot_name + az_name`.
        In the long representation the first line is the action on the
        Hamiltonian, the second line is `rot_name` and the third line
        is the unitary action as a matrix, if set.

        `rot_name` can be:
        - `1` for identity
        - `I` for inversion (in 1D mirror is the same as inversion)
        - `R(angle)` for 2D rotation
        - `R(angle, axis)` for 3D rotation (axis is not normalized)
        - `M(normal)` for mirror
        - `S(angle, axis)` for 3D rotoinversion (axis is not normalized)

        `az_name` can be:
        - `T` for time-reversal (antiunitary symmetry)
        - `P` for particle-hole (antiunitary antisymmetry)
        - `C` for chiral (unitary antisymmetry)
        - missing if the symmetry is unitary
    """

    def name_angle(theta, latex=False):
        frac = Fraction(theta / np.pi).limit_denominator(100)
        num, den = frac.numerator, frac.denominator
        if latex:
            if den == 1:
                angle = r'{}{}\pi'.format("-" if num < 0 else "",
                                         "" if abs(num) == 1 else abs(num))
            else:
                angle = r'{}\frac{{{}\pi}}{{{}}}'.format(
                            "-" if num < 0 else "",
                            "" if abs(num) == 1 else abs(num),
                            den)
        else:
            angle = '{}{}π{}'.format("-" if num < 0 else "",
                                     "" if abs(num) == 1 else abs(num),
                                     "" if den == 1 else ("/" + str(den)))
        return angle

    R = np.array(g.R).astype(float)
    if R.shape[0] == 1:
        if R[0, 0] == 1:
            rot_name = '1'
        else:
            rot_name = 'I'
    elif R.shape[0] == 2:
        if np.isclose(la.det(R), 1):
            # pure rotation
            theta = np.arctan2(R[1, 0], R[0, 0])
            if np.isclose(theta, 0):
                rot_name = '1'
            else:
                if latex:
                    rot_name = r'R\left({}\right)'.format(name_angle(theta, latex))
                else:
                    rot_name = 'R({})'.format(name_angle(theta))
        else:
            # mirror
            val, vec = la.eigh(R)
            assert allclose(val, [-1, 1]), R
            n = vec[:, 0]
            if latex:
                rot_name = r'M\left({}\right)'.format(_round_axis(n))
            else:
                rot_name = 'M({})'.format(_round_axis(n))
    elif R.shape[0] == 3:
        if np.isclose(la.det(R), 1):
            # pure rotation
            n, theta = rotation_to_angle(R)
            if np.isclose(theta, 0):
                rot_name = '1'
            else:
                if latex:
                    rot_name = (r'R\left({}, {}\right)'
                                .format(name_angle(theta, latex), _round_axis(n)))
                else:
                    rot_name = (r'R({}, {})'
                                .format(name_angle(theta, latex), _round_axis(n)))
        else:
            # rotoinversion
            n, theta = rotation_to_angle(-R)
            if np.isclose(theta, 0):
                # inversion
                rot_name = 'I'
            elif np.isclose(theta, np.pi):
                # mirror
                if latex:
                    rot_name = r'M\left({}\right)'.format(_round_axis(n))
                else:
                    rot_name = 'M({})'.format(_round_axis(n))
            else:
                # generic rotoinversion
                if latex:
                    rot_name = (r'S\left({}, {}\right)'
                                .format(name_angle(theta, latex), _round_axis(n)))
                else:
                    rot_name = ('S({}, {})'
                                .format(name_angle(theta, latex), _round_axis(n)))

    if full:
        if latex:
            name = r'U H(\mathbf{{k}}){} U^{{-1}} = {}H({}R\mathbf{{k}}) \\'
            name = name.format("^*" if g.conjugate else "", "-" if g.antisymmetry else "",
                               "-" if g.conjugate else "")
        else:
            name = '\nU⋅H(k){}⋅U^-1 = {}H({}R⋅k)\n'.format("*" if g.conjugate else "",
                                                         "-" if g.antisymmetry else "",
                                                         "-" if g.conjugate else "")
        name += 'R = {}'.format(rot_name) + (r'\\' if latex else '\n')
        if g.U is not None:
            if latex:
                Umat = _array_to_latex(np.round(g.U, 3))
                name += 'U = {}'.format(Umat)
            else:
                name += 'U = {}'.format(str(np.round(g.U, 3)).replace('\n', '\n    ')) +'\n\n'
    else:
        if g.conjugate and not g.antisymmetry:
            az_name = r" \mathcal{T}" if latex else "T"
        elif g.conjugate and g.antisymmetry:
            az_name = r" \mathcal{P}" if latex else "P"
        elif not g.conjugate and g.antisymmetry:
            az_name = r" \mathcal{C}" if latex else "C"
        else:
            az_name = ""
        name = (az_name if (rot_name == '1' and az_name != "")
                else rot_name + (" " if az_name is not "" else "") + az_name)
    return '$' + name + '$' if latex else name


def pretty_print_cgg(g, latex=False):
    """
    Return a human readable string representation of ContinuousGroupGenerator

    Parameters
    ----------

    g : ContinuousGroupGenerator
        Point group element to be represented.
    latex : bool (default False)
        Whether to output LateX formatted string.

    Returns
    -------
    name : string
        The first line is the action on the Hamiltonian, the following lines
        display the real space rotation as `R(ϕ)` for 2D rotation and
        `R(ϕ, axis)` for 3D rotation (axis is not normalized), and the conserved
        quantity as a matrix. If either of these is trivial, it is omitted.
        Note that a ContinuousGroupGenerator defines a continuous group
        of symmetries, so there is always a free parameter ϕ.
    """
    R, L = g.R, g.U
    # Find rotation axis
    if R is not None and allclose(R, np.zeros(R.shape)):
        R = None
    if L is not None and allclose(L, np.zeros(L.shape)):
        L = None

    if R is not None and R.shape[0] == 3:
        n = np.array([np.trace(l @ R) for l in L_matrices()]).real
        n /= la.norm(n)
        n = _round_axis(n)
        if latex:
            rot_name = r'R_{{\phi}} = R\left(\phi, {}\right)\\'.format(_round_axis(n))
        else:
            rot_name = '\nR_ϕ = R(ϕ, {})'.format(_round_axis(n))
    elif R is not None and R.shape[0] == 2:
        rot_name = r'R_{{\phi}} = R\left(\phi\right)\\' if latex else 'R_ϕ = R(ϕ)\n'
    else:
        rot_name = ''

    if L is not None:
        if latex:
            L_name = r'L = {} \\'.format(_array_to_latex(np.round(L, 3)))
        else:
            L_name = '\nL = {}'.format(str(np.round(L, 3)).replace('\n', '\n    '))

    if latex:
        if R is not None:
            name = r'{}H(\mathbf{{k}}){} = H(R_{{\phi}}\mathbf{{k}}) \\'
            name = name.format(r'e^{-i\phi L}' if L is not None else '',
                               r'e^{i\phi L}' if L is not None else '')
        else:
            name = r'\left[ H(\mathbf{{k}}), L \right] = 0 \\'
    else:
        if R is not None:
            name = '\n' + r'{}H(k){} = H(R_ϕ⋅k)'
            name = name.format(r'exp(-i ϕ L)⋅' if L is not None else '',
                               r'⋅exp(i ϕ L)' if L is not None else '')
        else:
            name = '\n[H(k), L] = 0'

    name += rot_name + L_name
    return '$' + name + '$' if latex else name


def rotation_to_angle(R):
    # Convert 3D rotation matrix to axis and angle
    assert allclose(R, R.real)
    n = np.array([1j * np.trace(L @ R) for L in L_matrices()]).real
    # Choose direction to minimize number of minus signs
    absn = la.norm(n) * np.sign(np.sum(n))
    if np.isclose(absn, 0):
        # n is zero for 2-fold rotation, find eigenvector with +1
        val, vec = la.eigh(R)
        assert np.isclose(val[-1], 1), R
        n = vec[:, -1]
        # Choose direction to minimize number of minus signs
        n /= np.sign(np.sum(n))
    else:
        n /= absn
    theta = np.arctan2(absn, np.trace(R).real - 1)
    return n, theta


def _round_axis(n):
    # Try to find integer axis
    for vec in product([-1, 0, 1], repeat=len(n)):
        if np.isclose(vec @ n, la.norm(vec)) and not np.isclose(la.norm(vec), 0):
            return np.array(vec, int)
    # otherwise round
    return np.round(n, 2)


def _array_to_latex(a, precision=2):
    """Returns a LaTeX bmatrix
    a: numpy array
    returns: LaTeX bmatrix as a string

    based on https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    rv = r'\begin{bmatrix}'
    for line in a:
        rv += np.array2string(line, precision=precision, separator='&', suppress_small=True)[1:-1] + r'\\'
    rv +=  r'\end{bmatrix}'
    return rv


## Predefined representations

def spin_matrices(s, include_0=False):
    """
    Construct spin-s matrices for any half-integer spin.

    Parameters
    ----------

    s : float or int
        Spin representation to use, must be integer or half-integer.
    include_0 : bool (default False)
        If `include_0` is True, S[0] is the identity, indices 1, 2, 3
        correspond to x, y, z. Otherwise indices 0, 1, 2 are x, y, z.

    Returns
    -------

    ndarray
        Sequence of spin-s operators in the standard spin-z basis.
        Array of shape `(3, 2*s + 1, 2*s + 1)`, or if `include_0` is True
        `(4, 2*s + 1, 2*s + 1)`.
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


def spin_rotation(n, s, roundint=False):
    """
    Construct the unitary spin rotation matrix for rotation specified by the
    vector n (in radian units) with angular momentum `s`, given by
    `U = exp(-i n⋅s)`.

    Parameters
    ----------

    n : iterable
        Rotation vector. Its norm is the rotation angle in radians.
    s : float or sequence of arrays
        Spin representation to use for the unitary action of the
        operator. If float is provided, it should be integer or half-integer
        specifying the spin representation in the standard basis, see `spin_matrices`.
        Otherwise a sequence of 3 arrays of identical square size must be provided
        representing 3 components of the angular momentum operator.
    roundint : bool (default False)
        If roundint is True, result is converted to integer tinyarray if possible.

    Returns
    -------
    U : ndarray
        Unitary spin rotation matrix of the same shape as the spin matrices or
        `(2*s + 1, 2*s + 1)`.
    """
    n = np.array(n)
    if isinstance(s, Number):
        s = spin_matrices(s)
    else:
        s = np.array(s)
    # Make matrix exponential for rotation representation
    U = la.expm(-1j * np.tensordot(n, s, axes=((0), (0))))
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

    Parameters
    ----------
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

    Returns
    -------
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


class PrettyList(list):
    """Subclass of list that displays its elements with latex."""
    def _repr_latex_(self):
        return (
            r'$'
            + r'\\'.join(i._repr_latex_().replace('$', '') for i in self)
            + r'$'
        )
