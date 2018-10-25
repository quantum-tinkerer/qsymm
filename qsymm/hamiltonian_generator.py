from collections import OrderedDict
import numpy as np
import sympy
import itertools as it
import scipy.linalg as la
from copy import deepcopy

from .linalg import matrix_basis, nullspace, sparse_basis, family_to_vectors, rref
from .linalg import matrix_basis, nullspace, sparse_basis, family_to_vectors, rref, allclose
from .model import Model, BlochModel, _commutative_momenta, e, I
from .groups import PointGroupElement, ContinuousGroupGenerator
from .groups import generate_group
from . import kwant_continuum


def continuum_hamiltonian(symmetries, dim, total_power, all_powers=None,
                          momenta=_commutative_momenta, sparse_linalg=False,
                          prettify=False, num_digits=10):
    """Generate a family of continuum Hamiltonians that satisfy symmetry constraints.

    Parameters:
    -----------------
    symmetries: iterable of PointGroupElement objects.
        An iterable of PointGroupElement objects, each describing a symmetry
        that the family should possess.
    dim: integer
        The number of spatial dimensions along which the Hamiltonian family is
        translationally invariant. Only the first `dim` entries in `all_powers` and
        `momenta` are used.
    total_power: integer or list of integers
        Allowed total powers of the momentum variables in the continuum Hamiltonian.
        If an integer is given, all powers below it are included as well.
    all_powers: list of integer or list of list of integers
        Allowed powers of the momentum variables in the continuum Hamiltonian listed
        for each spatial direction. If an integer is given, all powers below it are
        included as well. If a list of integers is given, only these powers are used.
    momenta : list of int or list of Sympy objects
        Indices of momenta from ['k_x', 'k_y', 'k_z'] or a list of names for the
        momentum variables. Default is ['k_x', 'k_y', 'k_z'].
    sparse_linalg : bool
        Whether to use sparse linear algebra. Using sparse solver can result in
        performance increase for large, highly constrained families,
        but not as well tested as the default dense version.
    prettify: boolean, default False
        Whether to make the basis pretty by rounding and basis change. For details
        see docstring of `make_basis_pretty`. May be numerically unstable.
    num_digits: integer, default 10
        Number of significant digits to which the basis is rounded using prettify.
        This argument is ignored if prettify = False.

    Returns:
    ---------------
    family: list
        A list of Model objects representing the family that
        satisfies the symmetries specified. Each Model object satisfies
        all the symmetries by construction.
    """

    if type(total_power) is int:
        max_power = total_power
        total_power = range(max_power + 1)

    # Generate a Hamiltonian family
    N = list(symmetries)[0].U.shape[0] # Dimension of Hamiltonian matrix
    momenta = momenta[:dim]
    # Symmetries do not mix the total degree of momenta. We can thus work separately at each
    # fixed degree.
    family = []
    for degree in total_power:
        # Make all momentum variables given the constraints on dimensions and degrees in the family
        momentum_variables = continuum_variables(dim, degree, all_powers=all_powers, momenta=momenta)
        degree_family = [Model({momentum_variable: matrix}, momenta=momenta)
                             for momentum_variable, matrix
                             in it.product(momentum_variables, matrix_basis(N))]
        degree_family = constrain_family(symmetries, degree_family, sparse_linalg=sparse_linalg)
        if prettify:
            family_size = len(degree_family)
            degree_family = make_basis_pretty(degree_family, num_digits=num_digits)
            assert family_size == len(degree_family), 'make_basis_pretty reduced the size of the family, \
                                                possibly due to numerical instability'
        family += degree_family
    return family


def continuum_pairing(symmetries, dim, total_power, all_powers=None, momenta=_commutative_momenta,
                      phases=None, ph_square=1, sparse_linalg=False,
                      prettify=False, num_digits=10):
    """Generate a family of continuum superconducting pairing functions that satisfy
    symmetry constraints.

    The specified symmetry operators should act on the normal state Hamiltonian, not
    in particle-hole space.

    Parameters:
    -----------------
    symmetries: iterable of PointGroupElement objects.
        An iterable of PointGroupElement objects, each describing a symmetry
        that the family should possess.
    dim: integer
        The number of spatial dimensions along which the Hamiltonian family is
        translationally invariant. Only the first `dim` entries in `all_powers` and
        `momenta` are used.
    total_power: integer or list of integers
        Allowed total powers of the momentum variables in the continuum Hamiltonian.
        If an integer is given, all powers below it are included as well.
    all_powers: list of integer or list of list of integers
        Allowed powers of the momentum variables in the continuum Hamiltonian listed
        for each spatial direction. If an integer is given, all powers below it are
        included as well. If a list of integers is given, only these powers are used.
    momenta: list of int or list of Sympy objects
        Indices of momenta from ['k_x', 'k_y', 'k_z'] or a list of names for the
        momentum variables. Default is ['k_x', 'k_y', 'k_z'].
    phases: iterable of numbers
        Phase factors to multiply the hole block of the symmetry operators in
        particle-hole space. By default, all phase factors are 1.
    ph_square: integer, either 1 or -1.
        Specifies whether the particle-hole operator squares to +1 or -1.
    sparse_linalg : bool
        Whether to use sparse linear algebra. Using sparse solver can result in
        performance increase for large, highly constrained families,
        but not as well tested as the default dense version.
    prettify: boolean, default False
        Whether to make the basis pretty by rounding and basis change. For details
        see docstring of `make_basis_pretty`. May be numerically unstable.
    num_digits: integer, default 10
        Number of significant digits to which the basis is rounded using prettify.
        This argument is ignored if prettify = False.
    Returns:
    ---------------
    family: list
        A list of Model objects representing the family that
        satisfies the symmetries specified. Each Model object satisfies
        all the symmetries by construction.
    """
    if type(total_power) is int:
        max_power = total_power
        total_power = range(max_power + 1)

    assert ph_square in (-1, 1), 'Particle-hole operator must square to +1 or -1.'
    if phases is None:
        phases = np.ones(len(symmetries))
    symmetries = deepcopy(symmetries)
    N = symmetries[0].U.shape[0]
    # Attach phases to symmetry operators and extend to BdG space
    for sym, phase in zip(symmetries, phases):
        if isinstance(sym, PointGroupElement):
            sym.U = la.block_diag(sym.U, phase*sym.U.conj())
        if isinstance(sym, ContinuousGroupGenerator):
            sym.U = la.block_diag(sym.U, -sym.U.conj() + phase*np.eye(N))
    # Build ph operator
    ph = np.array([[0, 1], [ph_square, 0]])
    ph = PointGroupElement(np.eye(dim), True, True, np.kron(ph, np.eye(N)))
    symmetries.append(ph)
    momenta = momenta[:dim]
    momentum_variables = continuum_variables(dim, total_power, all_powers=all_powers, momenta=momenta)
    # matrix basis for all matrices in off-diagonal blocks
    b0 = np.zeros((N, N))
    mbasis = [np.block([[b0, m], [m.T.conj(), b0]]) for m in matrix_basis(N)]
    mbasis.extend([np.block([[b0, 1j*m], [-1j*m.T.conj(), b0]]) for m in matrix_basis(N)])

    # Symmetries do not mix the total degree of momenta. We can thus work separately at each
    # fixed degree.
    family = []
    for degree in total_power:
        # Make all momentum variables given the constraints on dimensions and degrees in the family
        momentum_variables = continuum_variables(dim, degree, all_powers=all_powers, momenta=momenta)
        degree_family = [Model({momentum_variable: matrix}, momenta=momenta)
                             for momentum_variable, matrix
                             in it.product(momentum_variables, mbasis)]
        degree_family = constrain_family(symmetries, degree_family, sparse_linalg=sparse_linalg)
        if prettify:
            family_size = len(degree_family)
            degree_family = make_basis_pretty(degree_family, num_digits=num_digits)
            assert family_size == len(degree_family), 'make_basis_pretty reduced the size of the family, \
                                                possibly due to numerical instability'
        family += degree_family

    # Cast the pairing terms into new Model objects, to ensure that each object has the correct
    # shape.
    family = [Model({term: matrix[:N, N:] for term, matrix in monomial.items()}) for monomial in
              family]
    return [mon for mon in family if len(mon)]


def continuum_variables(dim, total_power, all_powers=None, momenta=_commutative_momenta):
    """Make a list of all linearly independent combinations of momentum
    variables with given total power.

    Parameters
    ----------------
    dim: integer
        The number of spatial dimensions along which the Hamiltonian family is
        translationally invariant. Only the first `dim` entries in `all_powers` and
        `momenta` are used.
    total_power: integer
        Allowed total power of the momentum variables in the continuum Hamiltonian.
    all_powers: list of integer or list of list of integers
        Allowed powers of the momentum variables in the continuum Hamiltonian listed
        for each spatial direction. If an integer is given, all powers below it are
        included as well. If a list of integers is given, only these powers are used.
    momenta : list of int or list of Sympy objects
        Indices of momenta from ['k_x', 'k_y', 'k_z'] or a list of names for the
        momentum variables. Default is ['k_x', 'k_y', 'k_z'].

    Returns
    ---------------
    A list of Sympy objects, representing the momentum variables that enter the Hamiltonian.
    """

    if all_powers is None:
        all_powers = [total_power] * dim

    if len(all_powers) < dim or len(momenta) < dim:
        raise ValueError('`all_powers` and `momenta` must be at least `dim` long.')
    # Only keep the first dim entries
    momenta = momenta[:dim]
    all_powers = all_powers[:dim]

    for i, power in enumerate(all_powers):
        if type(power) is int:
            all_powers[i] = range(power + 1)

    if dim == 0:
        return [kwant_continuum.sympify(1)]

    if all([type(i) is int for i in momenta]):
        momenta = [_commutative_momenta[i] for i in momenta]
    else:
        momenta = [kwant_continuum.make_commutative(k, k)
                        for k in momenta]

    # Generate powers for all terms
    powers = [p for p in it.product(*all_powers) if sum(p) == total_power]
    momentum_variables = [sympy.Mul(*[sympy.Pow(k, power) for k, power in zip(momenta, p)])
                          for p in powers]
    return momentum_variables


def round_family(family, num_digits=3):
    """Round the matrix coefficients of a family to specified significant digits.

    Parameters
    -----------
    family: iterable of Model objects that represents
        a family.
    num_digits: integer
        Number if significant digits to which the matrix coefficients are rounded.

    Returns
    ----------
    A list of Model objects representing the family, with
    the matrix coefficients rounded to num_digits significant digits.

    """
    return [member.around(num_digits) for member in family]


def hamiltonian_from_family(family, coeffs=None, nsimplify=True, tosympy=True):
    """Form a Hamiltonian from a Hamiltonian family by taking a linear combination
    of its elements.
    
    Parameters
    ----------
    family: iterable of Model or BlochModel objects
        List of terms in the Hamiltonian family.
    coeffs: list of sympy objects, optional
        Coefficients used to form the linear combination of
        terms in the family. Element n of coeffs multiplies
        member n of family. The default choice of the coefficients
        is c_n.
    nsimplify: bool
        Whether to use sympy.nsimplify on the output or not, which
        attempts to replace floating point numbers with simpler expressions,
        e.g. fractions.
    tosympy: bool
        Whether to convert the Hamiltonian to a sympy expression.
        If False, a Model or BlochModel object is returned instead,
        depending on the type of the Hamiltonian family.
    
    Returns
    -------
    ham: sympy.Matrix or Model/BlochModel object.
        The Hamiltonian, i.e. the linear combination of entries in family.
    
    """
    if coeffs is None:
        coeffs = list(sympy.symbols('c0:%d'%len(family), real=True))
    else:
        assert len(coeffs) == len(family), 'Length of family and coeffs do not match.'
    ham = sum(c * term for c, term in zip(coeffs, family))
    if tosympy:
        return ham.tosympy(nsimplify=nsimplify)
    else:
        return ham


def display_family(family, summed=False, coeffs=None, nsimplify=True):
    """Helper function to display a Hamiltonian family.
    Supports LaTeX display through Sympy in a Jupyter notebook, which may be enabled
    by running sympy.init_printing(print_builtin=True).

    Parameters
    -----------
    family: iterable of Model or BlochModel objects
        List of terms in a Hamiltonian family.
    summed: boolean
        Whether to display the Hamiltonian family by individual member (False),
        or as a sum over all members with expansion coefficients (True).
    coeffs: list of sympy objects, optional
        Coefficients used when combining terms in the family if summed is True.
    nsimplify: boolean
        Whether to use sympy.nsimplify on the output or not, which attempts to replace
        floating point numbers with simpler expressions, e.g. fractions.
    """

    if not summed:
        # print each member in the family separately
        for term in family:
            sterm = term.tosympy(nsimplify=nsimplify)
            display(sterm)
    else:
        # sum the family members multiplied by expansion coefficients
        display(hamiltonian_from_family(family, coeffs=coeffs,
                                        nsimplify=nsimplify))

def check_symmetry(family, symmetries, num_digits=None):
    """Check that a family satisfies symmetries. A symmetry is satisfied if all members of
    the family satisfy it.

    If the input family was rounded before hand, it is necessary to use
    specify the number of significant digits using num_digits, otherwise
    this check might fail.

    Parameters:
    ------------
    family: iterable of Model or BlochModel objects representing
        a family.
    symmetries: iterable representing the symmetries to check.
        If the family is a Hamiltonian family, symmetries is an iterable
        of PointGroupElement objects representing the symmetries
        to check.
    num_digits: integer
        In the case that the input family has been rounded, num_digits
        should be the number of significant digits to which the family
        was rounded.
    """

    for symmetry in symmetries:
        # Iterate over all members of the family
        for member in family:
            if isinstance(symmetry, PointGroupElement):
                if num_digits is None:
                    assert symmetry.apply(member) == member
                else:
                    assert symmetry.apply(member).around(num_digits) == member.around(num_digits)
            elif isinstance(symmetry, ContinuousGroupGenerator):
                # Continous symmetry if applying it returns zero
                assert symmetry.apply(member) == {}


def constrain_family(symmetries, family, sparse_linalg=False):
    """Apply symmetry constraints to a family.

    Parameters
    -----------
    symmetries: iterable of PointGroupElement objects, representing the symmetries
        that are used to constrain the Hamiltonian family.
    family: iterable of Model or BlochModel objects, representing the Hamiltonian
        family to which the symmetry constraints are applied.
    sparse_linalg : bool
        Whether to use sparse linear algebra. Using sparse solver can result in
        performance increase for large, highly constrained families,
        but not as well tested as the default dense version.

    Returns
    ----------
    family: iterable of Model or BlochModel objects, that represents the
        family with the symmetry constraints applied. """

    if not family:
        return family
    # Fix ordering
    family = list(family)
    symmetries = list(symmetries)
    # Check compatibility of family members and symmetries
    shape = family[0].shape
    momenta = family[0].momenta
    for term in family:
        assert term.shape == shape
        assert term.momenta == momenta
    for symmetry in symmetries:
        assert symmetry.U.shape == shape
        if symmetry.R is not None:
            assert symmetry.R.shape[0] == len(momenta)

    # Need all the linearly independent variables before and after
    # rotation to make the matrix of linear constraints.
    rotated_families = [[symmetry.apply(monomial) for monomial in family]
                                                  for symmetry in symmetries]

    # Get all variables and fix ordering
    all_variables = set()
    for member in it.chain(*rotated_families):
        all_variables |= member.keys()
    all_variables = list(all_variables)

    # Generate the matrix of symmetry constraints.
    constraint_matrices = []
    for i, symmetry in enumerate(symmetries):
        # In block space, each row is the constraint that the matrix coefficient to a linearly independent
        # monomial must vanish. The column index runs over expansion coefficients multiplying different
        # models.
        rotated_family = rotated_families[i]
        constraint_matrix = family_to_vectors(rotated_family, all_keys=all_variables).T
        if isinstance(symmetry, PointGroupElement):
            # Only need to subtract untransformed part for discrete symmetries,
            # continuous symmetry applies the infinitesimal generator and
            # the result should vanish.
            constraint_matrix -= family_to_vectors(family, all_keys=all_variables).T
        constraint_matrices.append(constraint_matrix)

    constraint_matrix = np.vstack(constraint_matrices)
    # If it is empty, there are no constraints
    if not np.any(constraint_matrix):
        return family
    # ROWS of this matrix are the basis vectors
    null_basis = nullspace(constraint_matrix, sparse=sparse_linalg)

    # We return a list of dictionary-like Model objects.
    # Each Model object represents one term in the Hamiltonian family,
    # where keys are the variables and values the matrix coefficients multiplying each variable.
    Hamiltonian_family = []
    for basis_vector in null_basis:
        family_member = sum([val * family[i] for i, val in enumerate(basis_vector)])
        # Eliminate entries that vanish
        if family_member:
            Hamiltonian_family.append(family_member)
    return Hamiltonian_family


def make_basis_pretty(family, num_digits=2):
    """Attempt to make a family more legible by reducing the
    number of nonzero entries in the matrix coefficients.

    Parameters
    -----------
    family: iterable of Model or BlochModel objects representing
        a family.
    num_digits: positive integer
        Number of significant digits that matrix coefficients are rounded to.

    This attempts to make the family more legible by prettifying a matrix,
    which is done by bringing it to reduced row-echelon form. This
    procedure may be numerically unstable, so this function should be used
    with caution. """

    # Return empty family for empty family
    if not family:
        return family
    # convert family to a set of row vectors
    basis_vectors = family_to_vectors(family)
    # Find the transformation that brings it to rref form
    _, S = rref(basis_vectors, return_S=True, rtol=10**(-num_digits))
    # Transform the model by S
    rfamily = []
    for vec in S:
        term = sum([val * family[i] for i, val in enumerate(vec)])
        # Eliminate entries that vanish
        if term:
            rfamily.append(term)
    return round_family(rfamily, num_digits)


def remove_duplicates(family, tol=1e-8):
    """Remove linearly dependent terms in Hamiltonian family using SVD.

    Parameters
    -----------
    family: iterable of Model or BlochModel objects representing
        a family.
    tol: float
        tolerance used in SVD when finding the span.

    Returns
    -------
    rfamily: list of Model or BlochModel objects representing
        the family with only linearly independent terms.
    """
    if not family:
        return family
    # Convert to vectors
    basis_vectors = family_to_vectors(family)
    # Find the linearly independent vectors
    _, basis_vectors = nullspace(basis_vectors.T, atol=tol, return_complement=True)
    rfamily = []
    for vec in basis_vectors:
        rfamily.append(sum([family[i] * c for i, c in enumerate(vec)]))
    return rfamily


def subtract_family(family1, family2, tol=1e-8, prettify=False):
    """Remove the linear span of family2 from the span of family1 using SVD.
    family2 must be a span of terms that are either inside the span of family1
    or orthogonal to it. This guarantees that projecting out family2 from family1
    results in a subfamily of family1.

    Parameters
    -----------
    family1, family2: iterable of Model or BlochModel objects
        Hamiltonian families.
    tol: float
        tolerance used in SVD when finding the span.

    Returns
    -------
    rfamily: list of Model or BlochModel objects representing
        family1 with the span of family2 removed.
    """
    if not family1 or not family2:
        return family1
    # Convert to vectors
    all_keys = set.union(*[set(term.keys()) for term in family1])
    all_keys |= set.union(*[set(term.keys()) for term in family2])
    all_keys = list(all_keys)
    basis1 = family_to_vectors(family1, all_keys=all_keys)
    basis2 = family_to_vectors(family2, all_keys=all_keys)
    # get the orthonormal basis for the span of basis2
    _, basis2 = nullspace(basis2, atol=tol, return_complement=True)
    # project out components in the span of basis2 from basis1
    projected_basis1 = basis1 - (basis1.dot(basis2.T.conj())).dot(basis2)
    # Check that projected_basis1 is a subspace of basis1.
    _, ort_basis1 = nullspace(basis1, atol=tol, return_complement=True)
    if not allclose((projected_basis1.dot(ort_basis1.T.conj())).dot(ort_basis1), projected_basis1, atol=tol):
        raise ValueError('Projecting onto the complement of family2 did not result in a subspace of family1')
    # Find the coefficients of linearly independent vectors
    _, projected_coeffs1 = nullspace(projected_basis1.T, atol=tol, return_complement=True)
    rfamily = []
    for vec in projected_coeffs1:
        rfamily.append(sum([family1[i] * c for i, c in enumerate(vec)]))
    if prettify:
        rfamily = make_basis_pretty(rfamily, num_digits=int(-np.log10(tol)))
    return rfamily


def symmetrize_monomial(monomial, symmetries):
    """Symmetrize monomial by averaging over all symmetry images under symmetries.

    Parameters:
    -----------
    monomial : Model or BlochModel object
        Hamiltonian term to be symmetrized
    symmetries : iterable of PointGroupElement objects
        Symmetries to use for symmetrization. `symmetries` must form a closed group.

    Returns:
    --------
    Model or BlochModel object
        Symmetrized term.
    """
    return sum([sym.apply(monomial) for sym in symmetries]) * (1/len(symmetries))


def bloch_family(hopping_vectors, symmetries, norbs, onsites=True,
                 symmetrize=True, prettify=True, num_digits=10,
                 bloch_model=False):
    """Generate a family of symmetric Bloch-Hamiltonians.

    Parameters:
    -----------
    hopping_vectors : list of tuples (a, b, vec)
        `a` and `b` are identifiers for the different sites (e.g. strings) of
        the unit cell, `vec` is the real space hopping vector. Vec may contain
        contain integers, sympy symbols, or floating point numbers.
    symmetries : list of PointGroupElement or ContinuousGroupGenerator
        Generators of the symmetry group. ContinuousGroupGenerators can only
        have onsite action as a lattice system cannot have continuous rotation
        invariance. It is assumed that the block structure of the unitary action
        is consistent with norbs, as returned by `symmetry_from_permutation`.
    norbs : OrderedDict : {site : norbs_site} or tuple of tuples ((site, norbs_site), )
        sites are ordered in the order specified, with blocks of size norbs_site
        corresponding to each site.
    onsites : bool
        Whether to include on-site terms consistent with the symmetry.
    symmetrize : bool
        Whether to use the symmetrization strategy. This does not require
        a full set of hoppings to start, all symmetry related hoppings
        are generated. Otherwise the constraining strategy is used, this does
        not generate any new hoppings beyond the ones specified and constrains
        coefficients to enforce symmetry.
    prettify: bool
        Whether to prettify the result. This step may be numerically unstable.
    num_digits: int, default 10
        Number of significant digits kept when prettifying.
    bloch_model: bool, default False
        Determines the return format of this function. If set to False, returns
        a list of Model objects. If True, returns a list of BlochModel objects.
        BlochModel objects are more suitable than Model objects if the hopping
        vectors include floating point numbers.

    Returns:
    --------
    family: list of Model or BlochModel objects
        A list of Model or BlochModel objects representing the family that
        satisfies the symmetries specified. Each object satisfies
        all the symmetries by construction.

    Notes:
    ------
    There is no need to specify the translation vectors, all necessary information
    is encoded in the symmetries and hopping vectors.

    In the generic case the Bloch-Hamiltonian produced is not Brillouin-zone periodic,
    instead it acquires a unitary transformation `W_G = delta_{ab} exp(i G (r_a - r_b))`
    where `G` is a reciprocal lattice vector, `a` and `b` are sites and `r_a` is the
    real space position of site `a`. If the lattice is primitive (there is only one
    site per unit cell), the hopping vectors are also translation vectors and the
    resulting Hamiltonian is BZ periodic.

    If `symmetrize=True`, all onsite unitary symmetries need to be explicitely
    specified as ContinuousGroupGenerators. Onsite PointGroupSymmetries (ones
    with R=identity) are ignored.
    
    If floating point numbers are used in the argument hopping_vectors, it is
    recommended to have this function return BlochModel objects instead of Model
    objects, by setting the bloch_model flag to True.
    """

    N = 0
    norbs = OrderedDict(norbs)
    ranges = dict()
    for a, n in norbs.items():
        ranges[a] = slice(N, N + n)
        N += n
    # Separate point group and conserved quantites
    pg = [g for g in symmetries if isinstance(g, PointGroupElement)]
    conserved = [g for g in symmetries if isinstance(g, ContinuousGroupGenerator)]
    if not all([(g.R is None or np.allclose(g.R, np.zeros_like(g.R))) for g in conserved]):
        raise ValueError('Bloch Hamiltonian cannot have continuous rotation symmetry.')

    # Check dimensionality
    dim = len(hopping_vectors[0][-1])
    assert all([len(hop[-1]) == dim for hop in hopping_vectors])

    # Add zero hoppings for onsites
    if onsites:
        hopping_vectors = deepcopy(hopping_vectors)
        hopping_vectors += [(a, a, [0] * dim) for a in norbs]

    family = []
    for a, b, vec in hopping_vectors:
        n, m = norbs[a], norbs[b]
        block_basis = np.eye(n*m, n*m).reshape((n*m, n, m))
        block_basis = np.concatenate((block_basis, 1j*block_basis))
        # Hopping direction in real space
        # Dot product with momentum vector
        phase = sum([coordinate * momentum for coordinate, momentum in
                     zip(vec, _commutative_momenta[:len(vec)])])
        factor = e**(I*phase)
        hopfamily = []
        for mat in block_basis:
            matrix = np.zeros((N, N), dtype=complex)
            matrix[ranges[a], ranges[b]] = mat
            term = Model({factor: matrix}, momenta=range(len(vec)))
            term = term + term.T().conj()
            hopfamily.append(term)
        # If there are conserved quantities, constrain the hopping, it is assumed that
        # conserved quantities do not mix different sites
        if conserved:
            hopfamily = constrain_family(conserved, hopfamily)
        family.extend(hopfamily)
    # Use BlochModel objects instead of Model.
    if bloch_model:
        family = [BlochModel(member, momenta=member.momenta) for
                  member in family]
    if symmetrize:
        # Make sure that group is generated while keeping track of unitary part.
        for g in pg:
            g._strict_eq = True
        pg = generate_group(set(pg))
        # Symmetrize every term and remove linearly dependent or zero ones
        family2 = []
        for term in family:
            term = symmetrize_monomial(term, pg).around(decimals=num_digits)
            if not term == {}:
                family2.append(term)
        family = remove_duplicates(family2, tol=10**(-num_digits))
    else:
        # Constrain the terms by symmetry
        family = constrain_family(pg, remove_duplicates(family))
    if prettify:
        family = make_basis_pretty(family, num_digits=num_digits)
    return family
