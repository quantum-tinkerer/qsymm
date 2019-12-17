import pytest
import warnings
import sympy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
import numpy as np

from pytest import raises
from ..model import _to_bloch_coeff, _commutative_momenta, Model, \
                    e, I, BlochModel, BlochCoeff, _symbol_normalizer
from ..linalg import allclose


k_x, k_y = _commutative_momenta[:2]
momenta = _commutative_momenta[:2]
c0, c1 = sympy.symbols('c0 c1', real=True)

def test_dense_algebra():
    # scalar models
    a = Model({1: 1, 'x': 3})
    assert a.shape == ()
    assert a.format is np.complex128
    assert a * 0 == 0 * a == a - a == 0
    assert 0 * a == a.zeros_like()
    assert 0 * a == {}
    assert a + 0 == a
    assert 0 + a == a
    assert a - a == {}
    assert a + a == 2 * a
    assert a + 2 == Model({1: 3, 'x': 3})
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
    assert c.format == np.ndarray
    assert c + 0 == c
    assert 0 + c == c
    assert c + 2 * vec == Model({1: 3, 'x': 3}) * vec
    assert 2 * vec + c == Model({1: 3, 'x': 3}) * vec
    # matrix models
    mat = np.ones((3, 3))
    d = b * mat
    assert d.shape == (3, 3)
    assert d.format == np.ndarray
    assert d * 0 == 0 * d == d - d == 0
    assert d + 0 == d
    assert 0 + d == d
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert c.T() @ d == 3 * a * b
    assert c.T() @ d @ c == 9 * a * a * b
    assert d @ d == 3 * b * d
    assert (d @ d).format == np.ndarray
    assert d + 2 * mat == Model({1: 4, 'x**2': 3}) * mat
    assert 2 * mat + d == Model({1: 4, 'x**2': 3}) * mat
    # numpy elemntwise multiplication
    assert d * c == a * b * mat
    assert d.trace() == 3 * b
    assert d.reshape(-1) == b * np.ones((9,))


def test_dense_algebra_bloch():
    # scalar models
    a = BlochModel({1: 1, 'e**(I*k_x)': 3})
    assert a.shape == ()
    assert a.format is np.complex128
    assert a * 0 == 0 * a
    assert 0 * a == a.zeros_like()
    assert 0 * a == {}
    assert a - a == {}
    assert a + a == 2 * a
    assert a + 2 == BlochModel({1: 3, 'e**(I*k_x)': 3})
    assert a / 2 == a * 0.5
    a += a
    assert a == BlochModel({1: 2, 'e**(I*k_x)': 6})
    a /= 2
    assert a == BlochModel({1: 1, 'e**(I*k_x)': 3})
    b = BlochModel({1: 2, 'e**(I*2*k_x)': 3})
    assert a + b == BlochModel({1: 3, 'e**(I*2*k_x)': 3, 'e**(I*k_x)': 3})
    assert a * b == BlochModel({1: 2, 'e**(I*k_x)': 6, 'e**(I*2*k_x)': 3, 'e**(I*3*k_x)': 9})
    assert (a - a) * b == {}
    assert (a - a) * (b - b) == {}
    # vector models
    vec = np.ones((3,))
    c = a * vec
    assert c.shape == (3,)
    assert c.format == np.ndarray
    assert c + 2 * vec == BlochModel({1: 3, 'e**(I*k_x)': 3}) * vec
    # matrix models
    mat = np.ones((3, 3))
    d = b * mat
    assert d.shape == (3, 3)
    assert d.format == np.ndarray
    assert d @ c == 3 * BlochModel({1: 2*vec, 'e**(I*k_x)': 6*vec,
                                    'e**(I*2*k_x)': 3*vec, 'e**(I*3*k_x)': 9*vec})
    assert c.T() @ d == 3 * a * b
    assert c.T() @ d @ c == 9 * a * a * b
    assert d @ d == 3 * b * d
    assert (d @ d).format == np.ndarray
    assert d + 2 * mat == BlochModel({1: 4, 'e**(I*2*k_x)': 3}) * mat
    # numpy elemntwise multiplication
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
    assert c.format is csr_matrix
    assert c + 2 * vec == Model({1: 3, 'x': 3}) * vec
    assert 2 * vec + c == Model({1: 3, 'x': 3}) * vec
    assert c + 0 == c
    assert 0 + c == c
    mat = csr_matrix(np.ones((3, 3)))
    d = b * mat
    assert d.shape == (3, 3)
    assert d.format is csr_matrix
    assert d + 0 == d
    assert 0 + d == d
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d).format is csr_matrix
    assert d + 2 * mat == Model({1: 4, 'x**2': 3}) * mat
    assert 2 * mat + d == Model({1: 4, 'x**2': 3}) * mat
    assert c.T() @ d == 3 * b * c.T()
    assert c.T() @ d @ c == 9 * a * a * b * np.eye(1)
    assert d.trace() == 3 * b
    assert d.reshape((1, 9)) @ np.ones((9,)) == 9 * b
    e = d @ np.ones((3,))
    assert e == 3 * b * np.ones((3,))
    assert e.format is np.ndarray

    # Test LinearOperator
    d = b * LinearOperator((3, 3), matvec=lambda v: mat @ v)
    c = a * np.ones((3, 1))
    assert d.shape == (3, 3)
    assert d.format is LinearOperator
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d).format is LinearOperator
    assert (d + 2 * LinearOperator((3, 3), matvec=lambda v: mat @ v)
            == Model({1: 4, 'x**2': 3}) * mat)
    assert d @ c == 3 * Model({1: 2*vec, 'x': 6*vec, 'x**2': 3*vec, 'x**3': 9*vec})
    assert (d @ c).format is np.ndarray
    assert c.T() @ (d @ c) == 9 * a * a * b


def test_sparse_algebra_bloch():
    # Test sparse matrices
    a = BlochModel({1: 1, 'e**(I*k_x)': 3})
    b = BlochModel({1: 2, 'e**(I*2*k_x)': 3})
    # sparse vector model
    vec = csr_matrix(np.ones((3, 1)))
    c = a * vec
    assert c.shape == (3, 1)
    assert c.format is csr_matrix
    assert c + 2 * vec == BlochModel({1: 3, 'e**(I*k_x)': 3}) * vec
    mat = csr_matrix(np.ones((3, 3)))
    d = b * mat
    assert d.shape == (3, 3)
    assert d.format is csr_matrix
    assert d @ c == 3 * BlochModel({1: 2*vec, 'e**(I*k_x)': 6*vec,
                                    'e**(I*2*k_x)': 3*vec, 'e**(I*3*k_x)': 9*vec})
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d).format is csr_matrix
    assert d + 2 * mat == BlochModel({1: 4, 'e**(I*2*k_x)': 3}) * mat
    assert c.T() @ d == 3 * b * c.T()
    assert c.T() @ d @ c == 9 * a * a * b * np.eye(1)
    assert d.trace() == 3 * b
    assert d.reshape((1, 9)) @ np.ones((9,)) == 9 * b
    e = d @ np.ones((3,))
    assert e == 3 * b * np.ones((3,))
    assert e.format is np.ndarray

    # Test LinearOperator
    d = b * LinearOperator((3, 3), matvec=lambda v: mat @ v)
    c = a * np.ones((3, 1))
    assert d.shape == (3, 3)
    assert d.format is LinearOperator
    assert d @ d == 3 * b * d
    assert (d @ d) @ c == 9 * b * b * c
    assert (d @ d).format is LinearOperator
    assert (d + 2 * LinearOperator((3, 3), matvec=lambda v: mat @ v)
            == BlochModel({1: 4, 'e**(I*2*k_x)': 3}) * mat)
    assert d @ c == 3 * BlochModel({1: 2*vec, 'e**(I*k_x)': 6*vec,
                                    'e**(I*2*k_x)': 3*vec, 'e**(I*3*k_x)': 9*vec})
    assert (d @ c).format is np.ndarray
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
    with raises(ValueError):
        bc = _to_bloch_coeff(key, momenta)

    key = (c1 + 3)*e**I*(k_x + k_y)
    with raises(ValueError):
        bc = _to_bloch_coeff(key, momenta)

    key = c0*k_y*e**(I*k_x)
    with raises(ValueError):
        bc = _to_bloch_coeff(key, momenta)

    key = sympy.sqrt(5)*e**(2*c0)*e**(I*(k_x/2 + np.sqrt(3)*k_y + c1*k_y))
    with raises(ValueError):
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
    with raises(NotImplementedError):
        prod = BC1 * 3.1


def test_Model():

    m = Model("sqrt(3) * e^(I*k_x + 2*I*k_y) + e^(-I*k_y)/2")
    keys= [_symbol_normalizer(e**(I*k_x + 2*I*k_y)),
           _symbol_normalizer(e**(-I*k_y))]
    assert all([key in keys for key in m.keys()])
    assert allclose(m[keys[0]], np.sqrt(3))
    assert allclose(m[keys[1]], 0.5)

    np.random.seed(seed=0)
    mat = np.random.rand(2,2)
    m = Model({np.sqrt(5)*e**(I*k_x) : mat}, normalize=True)
    assert allclose(m[list(m.keys())[0]], mat*np.sqrt(5))

    m1 = Model({k_x : 2.3})
    m2 = Model({k_y : 2.4, k_x : 1})
    assert not m1 == m2
    m2 += m1
    assert m2 == Model({k_x : 3.3, k_y : 2.4})
    m2 = Model(m1, momenta=['k_x'])
    with raises(ValueError):
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

    m2.momenta = (k_x,)
    with raises(ValueError):
        m1*m2

    m1 = Model({k_x : 4j, e**(I*k_y): 5})
    keys = [k_x, e**(-I*k_y)]
    m2 = m1.conj()
    assert all([key in keys for key in m2.keys()])
    assert m2[k_x] == m1[k_x].conj()
    assert m2[e**(-I*k_y)] == m1[e**(I*k_y)].conj()

def test_Model_locals():
    # Test that locals are treated properly
    ham1 = Model('alpha * sigma_z')
    ham2 = Model('alpha * sz', locals=dict(sz=np.diag([1, -1])))
    assert ham1 == ham2
    ham3 = Model('alpha * sz', locals=dict(sz='[[1, 0], [0, -1]]'))
    assert ham2 == ham3
    ham4 = Model('Hz', locals=dict(Hz='[[alpha, 0], [0, -alpha]]'))
    assert ham3 == ham4

def test_BlochModel():

    m = Model({e**(I*k_y): 3*np.eye(2), k_x : np.eye(2)}, normalize=True)
    with raises(ValueError):
        bm = BlochModel(m)
    m = Model({c1*e**(I*k_y): 3*np.eye(2), np.sqrt(3)*e**(I*k_x) : np.eye(2)},
              momenta=['k_x', 'k_y'], normalize=True)
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
    np.random.seed(seed=0)
    T = np.random.randint(10, size=(2,2))
    c0, c1 = sympy.symbols('c0 c1', real=True)
    Ham = Model({c0 * e**(-I*(k_x/2 + k_y ) + sympy.sqrt(2)) : T,
                 c1 * e**(I*(4*k_x + 3*k_y)) : 2*T}, momenta=['k_x', 'k_y'])
    u_1, u_2 = sympy.symbols('u_1 u_2', real=True)
    nHam = Ham.subs({k_x: u_1, k_y: 2*k_y + 1})
    assert nHam.momenta == (u_1, k_y)
    right_Ham = Model({c0*e**(-2*I*k_y - I*u_1/2): T * np.exp(np.sqrt(2)) * np.exp(-1j),
                       c1*e**(6*I*k_y + 4*I*u_1): 2 * T *np.exp(3j)},
                      momenta=[u_1, k_y])
    assert right_Ham.allclose(nHam), list((right_Ham - nHam).keys())

    nHam = Ham.subs(k_y, 1.5)
    assert nHam.momenta == (k_x,)
    right_Ham = Model({c0*e**(-I*k_x/2): T * np.exp(np.sqrt(2)) * np.exp(-1.5j),
                       c1*e**(4*I*k_x): 2 * T * np.exp(3*1.5j)},
                      momenta=[u_1, k_y])
    assert right_Ham.allclose(nHam), list((right_Ham - nHam).keys())

    nHam = Ham.subs(k_y, sympy.sqrt(3) + u_1)
    assert nHam.momenta == (k_x,)

    T = np.random.randint(10, size=(2,2))
    Ham = BlochModel({c0 * e**(-I*(k_x/2 + k_y )) : T,
                          c1 * e**(I*(4*k_x + 3*k_y)) : 2*T}, momenta=['k_x', 'k_y'])
    nHam = Ham.subs([(c0, 3), (c1, 2*u_1)])
    right_Ham = BlochModel({BlochCoeff(np.array([-0.5, -1]), sympy.numbers.One()): T * 3,
                       BlochCoeff(np.array([4, 3]), u_1): 2 * T * 2},
                      momenta=[k_x, k_y])

    assert right_Ham.allclose(nHam), list((right_Ham - nHam).keys())
