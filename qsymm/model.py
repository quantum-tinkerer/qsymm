import numpy as np
import tinyarray as ta
import scipy.linalg as la
import itertools as it
from copy import deepcopy
from numbers import Number
import sympy
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef
from collections import defaultdict, abc, UserDict
from .linalg import prop_to_id, allclose
from . import kwant_continuum


_commutative_momenta = [kwant_continuum.make_commutative(k, k)
           for k in kwant_continuum.momentum_operators]

e = kwant_continuum.sympify('e')
I = kwant_continuum.sympify('I')


def substitute_exponents(expr):
    """Substitute trignometric functions with exp.

    sin(X) -> (e^(I * X) - e^(-I * X)) / (2 * I)
    cos(X) -> (e^(I * X) + e^(-I * X)) / 2
    exp(X) -> e^X
    """

    subs = {}
    for f in expr.atoms(AppliedUndef, sympy.Function):
        # if more than one argument, we continue
        if len(f.args) > 1:
            continue
        else:
            arg = f.args[0]

        # if only one argument, we follow with subs
        if str(f.func) == 'sin':
            subs[f] = (e**(I * arg) - e**(-I * arg)) / (2 * I)

        if str(f.func) == 'cos':
            subs[f] = (e**(I * arg) + e**(-I * arg)) / 2

        if str(f.func) == 'exp':
            subs[f] = e**arg

    return expr.subs(subs).expand()


class HoppingCoeff(tuple):
    """Container for hopping coefficient in Model, equivalent to
    coeff * exp(I * hop.dot(k))."""

    def __new__(cls, hop, coeff):
        if not (isinstance(hop, np.ndarray) and isinstance(coeff, sympy.Expr)):
            raise ValueError('`hop` must be a 1D numpy array and `coeff` a sympy expression.')
        return super(HoppingCoeff, cls).__new__(cls, [hop, coeff])

    def __hash__(self):
        # only hash coeff
        return hash(self[1])

    def __eq__(self, other):
        hop1, coeff1 = self
        hop2, coeff2 = other
        # test equality of hop with allclose
        return allclose(hop1, hop2) and coeff1 == coeff2

    def __mul__(self, other):
        hop1, coeff1 = self
        if isinstance(other, sympy.Expr):
            return HoppingCoeff(hop1, coeff1 * other)
        elif isinstance(other, HoppingCoeff):
            hop2, coeff2 = other
            return HoppingCoeff(hop1 + hop2, coeff1 * coeff2)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        hop1, coeff1 = self
        if isinstance(other, sympy.Expr):
            return HoppingCoeff(hop1, other * coeff1)
        else:
            raise NotImplementedError

    def __deepcopy__(self, memo):
        hop, coeff = self
        return HoppingCoeff(deepcopy(hop), deepcopy(coeff))

    def tosympy(self, momenta):
        return sympy.exp(sympy.I * sum(ki * di for ki, di in zip(momenta, self[0])))


class Model(UserDict):

    # Make it work with numpy arrays
    __array_ufunc__ = None

    def __init__(self, hamiltonian=None, locals=None, momenta=[0, 1, 2]):
        """General class to store Hamiltonian families.
        Can be used to efficiently store any matrix valued function.
        Implements many sympy and numpy methods. Arithmetic operators are overloaded,
        such that `*` corresponds to matrix multiplication.

        Parameters
        ----------
        hamiltonian : str, SymPy expression or dict
            Symbolic representation of a Hamiltonian.  It is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            {sympy expression: np.ndarray} with all arrays the same square size
            and sympy expressions consisting purely of symbolic coefficients,
            no constant factors.
        locals : dict or ``None`` (default)
            Additional namespace entries for `~kwant_continuum.sympify`.  May be
            used to simplify input of matrices or modify input before proceeding
            further. For example:
            ``locals={'k': 'k_x + I * k_y'}`` or
            ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
        momenta : list of int or list of Sympy objects
            Indices of momenta the monomials depend on from 'k_x', 'k_y' and 'k_z'
            or a list of names for the momentum variables.
        """
        if all([type(i) is int for i in momenta]):
            self.momenta = [_commutative_momenta[i] for i in momenta]
        else:
            _momenta = [kwant_continuum.sympify(k) for k in momenta]
            self.momenta = [kwant_continuum.make_commutative(k, k)
                            for k in _momenta]
        if hamiltonian is None or isinstance(hamiltonian, abc.Mapping):
            keys = hamiltonian.keys()
            symbolic = all(isinstance(k, Basic) for k in keys)
            hopping = all(isinstance(k, HoppingCoeff) for k in keys)
            if not (symbolic or hopping):
                raise ValueError('All keys must have the same type (sympy expression or HoppingCoeff).')
            super().__init__(hamiltonian)
            # hopping is true if it is a Model that has HoppingCoeff as all of its keys and isn't empty
            self.hopping = hopping and (not hamiltonian == {})
            # Do not restructure if initialized with a dict.
            # self.restructure()
        else:

            hamiltonian = kwant_continuum.sympify(hamiltonian, locals=locals)

            if not isinstance(hamiltonian, sympy.matrices.MatrixBase):
                hamiltonian = sympy.Matrix([[hamiltonian]])

            hamiltonian = substitute_exponents(hamiltonian)

            free_parameters = list(hamiltonian.atoms(sympy.Symbol))
            gens = free_parameters + list(self.momenta)

            hamiltonian = kwant_continuum.make_commutative(hamiltonian, *gens)

            monomials = kwant_continuum.monomials(hamiltonian)

            monomials = {k: kwant_continuum.lambdify(v)()
                         for k, v in monomials.items()}

            # remove matrices == zeros
            monomials = {k: v for k, v in monomials.items()
                         if not np.allclose(v, 0)}

            self.data = monomials
            # This always generates a purely sympy Hamiltonian
            self.hopping = False

            # Restructure
            self.restructure()

        # Make sure every matrix has the same size
        if self == {}:
            self.shape = None
        else:
            shape = next(iter(self.values())).shape
            if not all([v.shape == shape for v in self.values()]):
                raise ValueError('All terms must have the same shape')
            self.shape = shape

    # Defaultdict functionality
    def __missing__(self, key):
        if self.shape is not None:
            return np.zeros(self.shape, dtype=complex)
        else:
            return None

    def __eq__(self, other):
        if not set(self) == set(other):
            return False
        for k, v in self.data.items():
            if not allclose(v, other[k]):
                return False
        return True

    def __add__(self, other):
        # Addition of monomials. It is assumed that both monomials are
        # structured correctly, every key is in standard form.
        # Define addition of 0 and {}
        if not other:
            result = self.copy()
        # If self is empty return other
        elif not self and isinstance(other, Model):
            result = other.copy()
        elif isinstance(other, Model):
            if self.momenta != other.momenta:
                raise ValueError("Model must have the same momenta")
            result = self.copy()
            for key, val in list(other.items()):
                if allclose(result[key], -val):
                    try:
                        del result[key]
                    except KeyError:
                        pass
                else:
                    result[key] += val
        else:
            raise NotImplementedError('Addition of monomials with type {} not supported'.format(type(other)))
        return result

    def __radd__(self, other):
        # Addition of monomials with other types.
        # If it evaluates to False, do nothing.
        if not other:
            return self.copy()
        else:
            raise NotImplementedError('Addition of monomials with type {} not supported'.format(type(other)))

    def __neg__(self):
        result = self.copy()
        for key, val in self.items():
            result[key] = -val
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        if isinstance(other, Number):
            if np.isclose(other, 0):
                result = self.zeros_like()
            else:
                result = self.copy()
                for key, val in result.items():
                    result[key] = other * val
        elif isinstance(other, Basic):
            result = Model({key * other: val for key, val in self.items()})
            result.momenta = self.momenta
        elif isinstance(other, np.ndarray):
            result = self.copy()
            for key, val in list(result.items()):
                prod = np.dot(val, other)
                if np.allclose(prod, 0):
                    del result[key]
                else:
                    result[key] = prod
            result.shape = np.dot(np.zeros(self.shape), other).shape
        elif isinstance(other, Model):
            if self.momenta != other.momenta:
                raise ValueError("Model must have the same momenta")
            result = sum(Model({k1 * k2: np.dot(v1, v2)})
                        for (k1, v1), (k2, v2) in it.product(self.items(), other.items()))
            result.momenta = list(set(self.momenta) | set(other.momenta))
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = Model({other * key: val for key, val in self.items()})
            result.momenta = self.momenta
        elif isinstance(other, np.ndarray):
            result = self.copy()
            for key, val in list(result.items()):
                prod = np.dot(other, val)
                if np.allclose(prod, 0):
                    del result[key]
                else:
                    result[key] = prod
            result.shape = np.dot(other, np.zeros(self.shape)).shape
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __repr__(self):
        result = ['{']
        for k, v in self.data.items():
            result.extend([str(k), ':\n', str(v), ',\n\n'])
        result.append('}')
        return "".join(result)

    def zeros_like(self):
        """Return an empty monomials object that inherits the size and momenta"""
        result = Model({})
        result.shape = self.shape
        result.momenta = self.momenta
        result.hopping = self.hopping
        return result

    def transform_symbolic(self, func):
        """Transform keys by applying func to all of them. Useful for
        symbolic substitutions, differentiation, etc. If key is a HoppingCoeff
        the substitution is only applied to the keys."""
        if self == {}:
            result = self.zeros_like()
        else:
            terms = []
            for key, val in self.items():
                if isinstance(key, sympy.Expr):
                    terms.append(Model({func(key): val}))
                else:
                    hop, coeff = key
                    terms.append(Model({HoppingCoeff(hop, func(coeff)): val}))
            result = sum(terms)
            # Remove possible duplicate keys that only differ in constant factors
            result.restructure()
            result.shape = self.shape
            result.momenta = self.momenta
        return result

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R"""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)

        if self.hopping:
            # do rotation on hopping vectors with transpose matrix
            R_T = np.array(R).T
            return Model({HoppingCoeff(R_T @ hop, coeff): val
                          for (hop, coeff), val in self.items()}, momenta=momenta)
        else:
            k_prime = R @ sympy.Matrix(momenta)
            rotated_subs = {k: k_prime for k, k_prime in zip(momenta, k_prime)}

            def trf(key):
                return key.subs(rotated_subs, simultaneous=True)

            return self.transform_symbolic(trf)

    def subs(self, *args, **kwargs):
        """Substitute symbolic expressions. See documentation of
        `sympy.Expr.subs()` for details."""
        return self.transform_symbolic(lambda x: x.subs(*args, **kwargs))

    def conj(self):
        """Complex conjugation"""
        if self.hopping:
            result = Model({HoppingCoeff(-hop, coeff.subs(sympy.I, -sympy.I)): val.conj()
                                                    for (hop, coeff), val in self.items()})
        else:
            result = Model({key.subs(sympy.I, -sympy.I): val.conj()
                                       for key, val in self.items()})
        result.momenta = self.momenta
        return result

    def T(self):
        """Transpose"""
        result = self.copy()
        for key, val in result.items():
            result[key] = val.T
        return result

    def value_list(self, key_list):
        """Return a list of the matrix coefficients corresponding to
        the keys in key_list"""
        return [self[key] for key in key_list]

    def around(self, decimals=3):
        """Return Model with matrices rounded to given number of decimals"""
        result = self.zeros_like()
        for key, val in self.items():
            val = np.around(val, decimals)
            if not np.allclose(val, 0):
                result[key] = val
        return result

    def restructure(self, atol=1e-8):
        """Clean internal data by:

        * splitting summands in keys
        * moving numerical factors to values
        * removing entries which values care np.allclose to zero
        """
        new_data = defaultdict(lambda: list())

        ### TODO: this loop seems quite inefficient. Maybe add an option
        # to skip it?
        for key, val in self.data.items():
            for summand in key.expand().powsimp(combine='exp').as_ordered_terms():
                factors = summand.as_ordered_factors()

                symbols, numbers = [], []
                for f in factors:
                    # This condition was previously
                    #    "if isinstance(f, sympy.numbers.Number):"
                    # but it didn't catch sqrt(2)
                    # then it was f.is_constant() but it's very slow
                    if f.is_number:
                        numbers.append(f)
                    else:
                        symbols.append(f)

                new_key = sympy.Mul(*symbols)
                new_val = complex(sympy.Mul(*numbers))  * val
                new_data[new_key].append(new_val)

        # translate list to single values
        new_data = {k: np.sum(np.array(v), axis=0)
                    for k, v in new_data.items()}

        # remove zero entries
        new_data = {k: v for k, v in new_data.items()
                    if not np.allclose(v, 0, atol=atol)}

        # overwrite internal data
        self.data = new_data

    def tosympy(self, nsimplify=False):
        # Return sympy representation of the term
        # If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas
        def keytosympy(key):
            if isinstance(key, Basic):
                return key
            else:
                return key.tosympy(self.momenta)

        if not nsimplify:
            return sympy.sympify(sum(keytosympy(key) * val for key, val in self.data.items()))
        else:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            return sympy.MatAdd(*[keytosympy(key) * sympy.Matrix(vnsimplify(val))
                              for key, val in self.data.items()]).doit()

    def copy(self):
        return deepcopy(self)
