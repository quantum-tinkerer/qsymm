import pytest
import warnings
import sympy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
import numpy as np

from pytest import raises
from ..model import _to_bloch_coeff, _commutative_momenta, Model, \
                    e, I, BlochModel, BlochCoeff
from ..linalg import allclose


k_x, k_y = _commutative_momenta[:2]
momenta = _commutative_momenta[:2]
c0, c1 = sympy.symbols('c0 c1', real=True)

def test_dense_algebra():
    # scalar models
    a = Model({1: 1, 'x': 3})
    assert a.shape == ()
    assert a._isarray == False
    assert a * 0 == 0 * a
    assert 0 * a == a.zeros_like()
    assert 0 * a == {}
    assert a - a == {}
    assert a + a == 2 * a
    assert a / 2 == a * 0.5
    a += a
    assert a == Model({1: 2, 'x': 6})
    a /= 2
    assert a == Model({1: 1, 'x': 3})
    b = Model({1: 2, 'x**2': 3})
    assert a + b == Model({1: 3, 'x': 3, 'x**2': 3})
    assert a * b == Model({1: 2, 'x': 6, 'x**2': 3, 'x**3': 9})
    assert (a - a) * b == {}
    assert (a - a) * (b - b) == {}
    # vector models
    vec = np.ones((3,))
    c = a * vec
    assert c.shape == (3,)
    assert c._isarray == True
    # matrix models
    mat = np.ones((3, 3))
    d = b * mat
    assert d.shape == (3, 3)
    assert d._isarray == True
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert c.T() @ d == 3 * a * b
    assert c.T() @ d @ c == 9 * a * a * b
    assert d @ d == 3 * b * d
    assert (d @ d)._isarray == True
    # numpy broadcasting rules apply
    assert d + c == (a + b) * mat
    assert d * c == a * b * mat
    assert d.trace() == 3 * b
    assert d.reshape(-1) == b * np.ones((9,))


def test_sparse_algebra():
    # Test sparse matrices
    a = Model({1: 1, 'x': 3})
    b = Model({1: 2, 'x**2': 3})
    # sparse vector model
    vec = csr_matrix(np.ones((3, 1)))
    c = a * vec
    assert c.shape == (3, 1)
    assert c._isarray == False
    mat = csr_matrix(np.ones((3, 3)))
    d = b * mat
    assert d.shape == (3, 3)
    assert d._isarray == False
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d)._isarray == False
    assert c.T() @ d == 3 * b * c.T()
    assert c.T() @ d @ c == 9 * a * a * b * np.eye(1)
    assert d.trace() == 3 * b
    assert d.reshape((1, 9)) @ np.ones((9,)) == 9 * b
    e = d @ np.ones((3,))
    assert e == 3 * b * np.ones((3,))
    assert e._isarray == True

    # Test LinearOperator
    d = b * LinearOperator((3, 3), matvec=lambda v: mat @ v)
    c = a * np.ones((3, 1))
    assert d.shape == (3, 3)
    assert d._isarray == False
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d)._isarray == False
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert (d @ c)._isarray == True
    assert c.T() @ (d @ c) == 9 * a * a * b


def test_to_bloch_coeff():

    key = sympy.sqrt(5)*e**(2*c0)*e**(I*(k_x/2 + np.sqrt(3)*k_y + c1))
    bc = _to_bloch_coeff(key, momenta)
    assert allclose(bc[0], np.array([1./2, np.sqrt(3)]))
    assert sympy.simplify(bc[1]) == sympy.simplify(sympy.sqrt(5)*e**(2*c0 + I*c1))

    key = sympy.sqrt(3)*c0
    bc = _to_bloch_coeff(key, momenta)
    assert allclose(bc[0], [0, 0])
    assert bc[1] == sympy.sqrt(3)*c0

    key = e**(2*k_x)
    with raises(AssertionError, message="Momenta in hopping exponentials must have the prefactor i."):
        bc = _to_bloch_coeff(key, momenta)

    key = (c1 + 3)*e**I*(k_x + k_y)
    with raises(ValueError, message="Coefficients cannot be sums of terms."):
        bc = _to_bloch_coeff(key, momenta)

    key = c0*k_y*e**(I*k_x)
    with raises(AssertionError, message="All momentum dependence should be in the hopping."):
        bc = _to_bloch_coeff(key, momenta)

    key = sympy.sqrt(5)*e**(2*c0)*e**(I*(k_x/2 + np.sqrt(3)*k_y + c1*k_y))
    with raises(AssertionError, message="Sympy coefficients not allowed in the real space hopping direction."):
        bc = _to_bloch_coeff(key, momenta)


def test_BlochCoeff():

    BC1 = BlochCoeff(np.array([2, 1]), c0)
    BC2 = BlochCoeff(np.array([3, -7]), c1)
    prod = BC1 * BC2
    assert allclose(prod[0], BC1[0] + BC2[0])
    assert prod[1] == BC1[1]*BC2[1]

    prod = BC1 * sympy.sqrt(3) * c1
    assert allclose(prod[0], BC1[0])
    assert prod[1] == sympy.sqrt(3)*BC1[1] * c1
    with raises(NotImplementedError,
                message="Only right multiplication with sympy expressions supported."):
        prod = BC1 * 3.1


def test_Model():

    m = Model("sqrt(3) * e^(I*k_x + 2*I*k_y) + e^(-I*k_y)/2")
    keys= [e**(I*k_x + 2*I*k_y), e**(-I*k_y)]
    assert all([key in keys for key in m.keys()])
    assert allclose(m[keys[0]], np.sqrt(3))
    assert allclose(m[keys[1]], 0.5)

    mat = np.random.rand(2,2)
    m = Model({np.sqrt(5)*e**(I*k_x) : mat}, restructure_dict=True)
    assert allclose(m[list(m.keys())[0]], mat*np.sqrt(5))

    m1 = Model({k_x : 2.3})
    m2 = Model({k_y : 2.4, k_x : 1})
    assert not m1 == m2
    m2 += m1
    assert m2 == Model({k_x : 3.3, k_y : 2.4})
    m2 = Model(m1, momenta = [0])
    with raises(ValueError,
                message="Only addition of Models with identical momenta allowed."):
        m2 + m1
    m2 = m1
    assert m1 == m2

    m2 = Model({k_y : 2.4, k_x : 1})
    prod = m1 * m2
    keys = [k_x**2, k_x*k_y]
    assert all([key in keys for key in prod.keys()])
    assert prod[keys[0]] == m1[k_x] * m2[k_x]
    assert prod[keys[1]] == m1[k_x] * m2[k_y]
    assert (m1 * np.array([2]))[k_x] == 2*m1[k_x]

    m2.momenta = [k_x]
    with raises(ValueError,
                message="Only multiplication of Models with identical momenta allowed."):
        m1*m2

    m1 = Model({k_x : 4j, e**(I*k_y): 5})
    keys = [k_x, e**(-I*k_y)]
    m2 = m1.conj()
    assert all([key in keys for key in m2.keys()])
    assert m2[k_x] == m1[k_x].conj()
    assert m2[e**(-I*k_y)] == m1[e**(I*k_y)].conj()


def test_BlochModel():

    m = Model({e**(I*k_y): 3*np.eye(2), k_x : np.eye(2)}, restructure_dict=True)
    with raises(AssertionError, message="All momentum dependence should be in the hopping."):
        bm = BlochModel(m)
    m = Model({c1*e**(I*k_y): 3*np.eye(2), np.sqrt(3)*e**(I*k_x) : np.eye(2)},
              momenta=[0, 1], restructure_dict=True)
    bm = BlochModel(m)
    keys = [BlochCoeff(np.array([0, 1]), c1), BlochCoeff(np.array([1, 0]), sympy.numbers.One())]
    assert all([key in keys for key in bm.keys()])
    assert allclose(bm[keys[0]], m[c1*e**(I*k_y)])
    assert allclose(bm[keys[1]], m[e**(I*k_x)])

    bm2 = (bm + bm).tomodel(nsimplify=True)
    keys2 = [c1*e**(I*k_y), e**(I*k_x)]
    assert all([key in keys2 for key in bm2.keys()])
    assert allclose(bm2[keys2[0]], 2*bm[keys[0]])
    assert allclose(bm2[keys2[1]], 2*bm[keys[1]])


def test_Model_subs():
    T = np.random.randint(10, size=(2,2))
    c0, c1 = sympy.symbols('c0 c1', real=True)
    Ham = Model({c0 * e**(-I*(k_x/2 + k_y ) + sympy.sqrt(2)) : T,
                 c1 * e**(I*(4*k_x + 3*k_y)) : 2*T}, momenta=[0, 1])
    u_1, u_2 = sympy.symbols('u_1 u_2', real=True)
    nHam = Ham.subs({k_x: u_1, k_y: 2*k_y + 1})
    assert nHam.momenta == [u_1, k_y]
    right_keys = [sympy.simplify(c0*e**(-2*I*k_y - I*u_1/2)),
                  sympy.simplify(c1*e**(6*I*k_y + 4*I*u_1))]
    nHam_keys = [sympy.simplify(key) for key in nHam.keys()]
    assert all([key in nHam_keys for key in right_keys])
    new_keys = list(nHam.keys())
    old_keys = list(Ham.keys())
    assert allclose(nHam[new_keys[0]], Ham[old_keys[0]]*np.exp(np.sqrt(2))*np.exp(-1j))
    assert allclose(nHam[new_keys[1]], Ham[old_keys[1]]*np.exp(3j))


    nHam = Ham.subs(k_y, 1.5)
    assert nHam.momenta == [k_x]
    right_keys = [sympy.simplify(c0*e**(-I*k_x/2)),
                  sympy.simplify(c1*e**(4*I*k_x))]
    nHam_keys = [sympy.simplify(key) for key in nHam.keys()]
    assert all([key in nHam_keys for key in right_keys])
    new_keys = list(nHam.keys())
    old_keys = list(Ham.keys())
    assert allclose(nHam[new_keys[0]], Ham[old_keys[0]]*np.exp(np.sqrt(2))*np.exp(-1.5j))
    assert allclose(nHam[new_keys[1]], Ham[old_keys[1]]*np.exp(3*1.5j))


    nHam = Ham.subs(k_y, sympy.sqrt(3) + u_1)
    assert nHam.momenta == [k_x]

    Ham = BlochModel({c0 * e**(-I*(k_x/2 + k_y )) : T,
                          c1 * e**(I*(4*k_x + 3*k_y)) : 2*T}, momenta=[0, 1])
    nHam = Ham.subs([(c0, 3), (c1, 2*u_1)])
    right_keys = [BlochCoeff(np.array([-0.5, -1]), sympy.numbers.One()),
                    BlochCoeff(np.array([4, 3]), u_1)]
    old_keys = list(Ham.keys())
    assert all([key in list(nHam.keys()) for key in right_keys])
    assert allclose(nHam[right_keys[0]], Ham[old_keys[0]]*3)
    assert allclose(nHam[right_keys[1]], Ham[old_keys[1]]*2)

