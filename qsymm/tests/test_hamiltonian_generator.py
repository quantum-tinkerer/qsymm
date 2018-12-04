import pytest
import sympy
import numpy as np
import scipy.linalg as la

from .. import kwant_rmt
from ..hamiltonian_generator import continuum_hamiltonian, check_symmetry, \
     bloch_family, make_basis_pretty, constrain_family, continuum_variables, \
     continuum_pairing, remove_duplicates, subtract_family
from ..groups import PointGroupElement, Model
from ..model import _commutative_momenta
from ..linalg import nullspace, family_to_vectors


def test_check_symmetry():
    """Test discrete symmetries for the Hamiltonian generator without momentum
    dependence, and that antiunitary and anticommuting symmetries are correctly
    treated in check_symmetry. """

    # No momentum dependence
    dim = 0
    total_power = 0
    n = 8
    R = np.eye(dim, dtype=int)

    for sym in kwant_rmt.sym_list:
        symmetries = []
        if kwant_rmt.p(sym):
            p_mat = np.array(kwant_rmt.h_p_matrix[sym])
            p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
            symmetries.append(PointGroupElement(R, True, True, p_mat))
        if kwant_rmt.t(sym):
            t_mat = np.array(kwant_rmt.h_t_matrix[sym])
            t_mat = np.kron(np.identity(n // len(t_mat)), t_mat)
            symmetries.append(PointGroupElement(R, True, False, t_mat))
        if kwant_rmt.c(sym):
            c_mat = np.kron(np.identity(n // 2), np.diag([1, -1]))
            symmetries.append(PointGroupElement(R, False, True, c_mat))

        if len(symmetries):
            Hamiltonian_family = continuum_hamiltonian(symmetries, dim, total_power)
            # Check that the symmetries hold.
            check_symmetry(Hamiltonian_family, symmetries)
            # Also manually check that the family has the symmetries, to test
            # the function check_symmetry.
            # Check several random linear combinations of all family members.
            for _ in range(5):
                ham = sum([np.random.rand()*matrix for member in Hamiltonian_family
                           for matrix in member.values()])
                # Iterate over all symmetries
                for symmetry in symmetries:
                    Left = ham.dot(symmetry.U)
                    if symmetry.conjugate: # Symmetry is antiunitary
                        Right = symmetry.U.dot(ham.conj())
                    else:  # Symmetry is unitary
                        Right = symmetry.U.dot(ham)
                    if symmetry.antisymmetry: # Symmetry anticommutes
                        assert np.allclose(Left + Right, 0)
                    else: # Symmetry commutes
                        assert np.allclose(Left - Right, 0)


def test_bloch_generator():
    """Square lattice with time reversal and rotation symmetry, such that all hoppings are real. """
    # Time reversal
    trU = np.eye(2)
    trR = sympy.Matrix(np.eye(2, dtype=int))
    trS = PointGroupElement(trR, True, False, trU)
    # 2-fold rotation
    rotU = np.eye(2)
    sphi = sympy.pi
    rotR = sympy.Matrix([[sympy.cos(sphi), -sympy.sin(sphi)],
                         [sympy.sin(sphi), sympy.cos(sphi)]])
    rotS = PointGroupElement(rotR, False, False, rotU)
    symmetries = [rotS, trS]
    
    # With integer hopping, output is list of Model
    hopping_vectors = [(0, 0, np.array([0, 1])), (0, 0, np.array([1, 0]))]
    norbs = {0: 2}
    family = bloch_family(hopping_vectors, symmetries, norbs, onsites=False)
    assert len(family) == 6, 'Incorrect number of members in the family'
    check_symmetry(family, symmetries)
    pretty = make_basis_pretty(family, num_digits=3)
    check_symmetry(pretty, symmetries)
    # All members should be real given the constraints.
    assert all([np.allclose(value, value.conj()) for member in pretty for value in member.values()])
    assert all([np.allclose(value, value.conj()) for member in family for value in member.values()])
    # Constraining the family again should have no effect
    again = constrain_family(symmetries, family)
    check_symmetry(again, symmetries)
    for member in family:
        assert any([member == other for other in again])
        
    # With floating point hopping, output is list of BlochModel
    hopping_vectors = [(0, 0, np.array([0, 0.5])), (0, 0, np.array([0.5, 0]))]
    norbs = {0: 2}
    family = bloch_family(hopping_vectors, symmetries, norbs, onsites=False, bloch_model=True)
    assert len(family) == 6, 'Incorrect number of members in the family'
    check_symmetry(family, symmetries)
    pretty = make_basis_pretty(family, num_digits=3)
    check_symmetry(pretty, symmetries)
    # All members should be real given the constraints.
    assert all([np.allclose(value, value.conj()) for member in pretty for value in member.values()])
    assert all([np.allclose(value, value.conj()) for member in family for value in member.values()])
    # Constraining the family again should have no effect
    again = constrain_family(symmetries, family)
    check_symmetry(again, symmetries)
    for member in family:
        assert any([member == other for other in again])



def test_continuum_variables():
    dim = 2
    total_power = 2

    # Test default arguments
    momenta = _commutative_momenta
    terms = continuum_variables(dim, total_power)
    result = [momenta[0]**0, momenta[0], momenta[1], momenta[0]**2, momenta[1]**2, momenta[1]*momenta[0]]
    assert all([term in result for term in terms]), 'Incorrect momentum terms returned by continuum_variables.'

    # Limit some powers of k
    all_powers = [1, 2]
    terms = continuum_variables(dim, total_power, all_powers=all_powers)
    result = [momenta[0]**0, momenta[0], momenta[1], momenta[1]**2, momenta[1]*momenta[0]]
    assert all([term in result for term in terms]), 'Incorrect limitations on powers in continuum_variables.'
    all_powers = [[1], [1, 3]]
    momenta = _commutative_momenta
    result = [momenta[1]*momenta[0]]
    terms = continuum_variables(dim, total_power, all_powers=all_powers)
    assert all([term in result for term in terms]), 'Incorrect limitations on powers in continuum_variables.'

    # Test different momenta
    dim = 2
    total_power = 4
    all_powers = [[1], [1, 3]]
    momenta = [sympy.Symbol('k_1'), sympy.Symbol('k_a')]
    result = [momenta[1]*momenta[0], momenta[0]*momenta[1]**3]
    terms = continuum_variables(dim, total_power, all_powers=all_powers, momenta=momenta)
    assert all([term in result for term in terms]), 'New momentum variables treated incorrectly.'


def test_pairing_generator():

    mU = np.array([[0.0, 1.0j],
                   [1.0j, 0.0]])
    mR = np.array([[-1]])

    # Use the pairing generator
    ph_squares = [1, -1]
    phases = np.hstack((np.exp(1j*np.random.rand(1)), np.array([1])))
    for ph_square in ph_squares:
        for phase in phases:
            mS = PointGroupElement(mR, False, False, mU)
            symmetries = [mS]
            dim = 1  # Momenta along x
            total_power = 1 # Maximum power of momenta
            from_pairing = continuum_pairing(symmetries, dim, total_power, ph_square=ph_square,
                                             phases=(phase, ), prettify=False)
            # Use the continuum Hamiltonian generator
            # Mirror
            mS = PointGroupElement(mR, False, False, mU)
            symmetries = [mS]
            # Extend the symmetry to particle-hole space
            N = mS.U.shape[0]
            mS.U = la.block_diag(mS.U, phase*mS.U.conj())
            # Build ph operator
            ph = np.array([[0, 1], [ph_square, 0]])
            ph = PointGroupElement(np.eye(dim), True, True, np.kron(ph, np.eye(N)))
            symmetries.append(ph)
            # Pick out pairing blocks
            cont_ham = continuum_hamiltonian(symmetries, dim, total_power)
            from_ham = [Model({term: matrix[:N, N:] for term, matrix in monomial.items()}) for monomial in
                          cont_ham]
            from_ham = [mon for mon in from_ham if len(mon)]
            from_ham = remove_duplicates(from_ham)

            assert len(from_pairing) == len(from_ham)
            # Combine into one redundant family, and then remove the redundancies.
            remove_dupls = remove_duplicates(from_pairing + from_ham)
            assert len(remove_dupls) == len(from_ham)


def test_subtract_family():
    paulis = [np.eye(2),
              np.array([[0, 1], [1, 0]]),
              np.array([[0, -1j], [1j, 0]]),
              np.array([[1, 0], [0, -1]])]
    One = _commutative_momenta[0]**0
    fam1 = [Model({One: pmat*(0.5 + np.random.rand())}) for pmat in paulis]
    fam2 = [Model({One: paulis[2] + paulis[3]}), Model({One: paulis[2] - paulis[3]})]
    result = subtract_family(fam1, fam2)
    correct = [Model({One: pmat}) for pmat in paulis[:2]]  # Correct span
    # Check that the correct result and the returned result have the same span
    null_mat = np.hstack((family_to_vectors(result).T, -family_to_vectors(correct).T))
    null_basis = nullspace(null_mat)
    assert null_basis.shape[0] == len(correct)


def test_pretty_basis():
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    One = _commutative_momenta[0]**0
    fam = [Model({One: sx*(0.5 + np.random.rand()) + sz*(0.5 + np.random.rand())}),
           Model({One: sx*(0.5 + np.random.rand()) - sz*(0.5 + np.random.rand())})]
    pfam = make_basis_pretty(fam, num_digits=3)
    assert len(pfam) == len(fam), 'Basis sparsification failed, returned a smaller basis.'
    # The sparse basis should be sx and sz
    for member in pfam:
        mat = list(member.values())[0]
        if np.isclose(mat[0, 0], 0):
            assert np.allclose(mat/mat[0, 1], sx), 'Sparsification failed.'
        else:
            assert np.allclose(mat/mat[0, 0], sz), 'Sparsification failed.'

