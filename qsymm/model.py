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


class BlochCoeff(tuple):

    def __new__(cls, hop, coeff):
        """Container for Bloch coefficient in Model, in the form of
        `(hop, coeff)`, equivalent to `coeff * exp(I * hop.dot(k))`."""
        if not (isinstance(hop, np.ndarray) and isinstance(coeff, sympy.Expr)):
            raise ValueError('`hop` must be a 1D numpy array and `coeff` a sympy expression.')
        if isinstance(coeff, sympy.add.Add):
            raise ValueError('`coeff` must be a single term with no sum.')
        return super(BlochCoeff, cls).__new__(cls, [hop, coeff])

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
            return BlochCoeff(hop1, coeff1 * other)
        elif isinstance(other, BlochCoeff):
            hop2, coeff2 = other
            return BlochCoeff(hop1 + hop2, coeff1 * coeff2)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        hop1, coeff1 = self
        if isinstance(other, sympy.Expr):
            return BlochCoeff(hop1, other * coeff1)
        else:
            raise NotImplementedError

    def __deepcopy__(self, memo):
        hop, coeff = self
        return BlochCoeff(deepcopy(hop), deepcopy(coeff))

    def tosympy(self, momenta, nsimplify=False):
        hop, coeff = self
        if nsimplify:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            hop = vnsimplify(hop)
        return coeff * e**(sum(I * ki * di for ki, di in zip(momenta, hop)))


class Model(UserDict):

    # Make it work with numpy arrays
    __array_ufunc__ = None

    def __init__(self, hamiltonian={}, locals=None, momenta=[0, 1, 2]):
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
        self.momenta = _find_momenta(momenta)
        if hamiltonian == {} or isinstance(hamiltonian, abc.Mapping):
            if not all(isinstance(k, Basic) for k in hamiltonian.keys()):
                raise ValueError('All keys must be sympy expressions.')
            # Initialize as dict
            super().__init__(hamiltonian)
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

        # Restructure
        # Clean internal data by:
        # * splitting summands in keys
        # * moving numerical factors to values
        # * removing entries which values care np.allclose to zero
        new_data = defaultdict(lambda: list())
        ### TODO: this loop seems quite inefficient. Maybe add an option
        # to skip it?
        for key, val in self.data.items():
            for summand in key.expand().powsimp(combine='exp').as_ordered_terms():
                factors = summand.as_ordered_factors()
                symbols, numbers = [], []
                for f in factors:
                    # This catches sqrt(2) and much faster than f.is_constant()
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
                    if not allclose(v, 0)}
        # overwrite internal data
        self.data = new_data

        self.shape = _find_shape(self.data)

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
        # Addition of Models. It is assumed that both Models are
        # structured correctly, every key is in standard form.
        # Define addition of 0 and {}
        if not other:
            result = self.copy()
        # If self is empty return other
        elif not self and isinstance(other, type(self)):
            result = other.copy()
        elif isinstance(other, type(self)):
            if self.momenta != other.momenta:
                raise ValueError("Can only add Models with the same momenta")
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
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))
        return result

    def __radd__(self, other):
        # Addition of monomials with other types.
        # If it evaluates to False, do nothing.
        if not other:
            return self.copy()
        else:
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))

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
            result = type(self)({key * other: val for key, val in self.items()})
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
        elif isinstance(other, type(self)):
            if self.momenta != other.momenta:
                raise ValueError("Can only multiply Models with the same momenta")
            result = sum(type(self)({k1 * k2: np.dot(v1, v2)})
                        for (k1, v1), (k2, v2) in it.product(self.items(), other.items()))
            result.momenta = list(set(self.momenta) | set(other.momenta))
        else:
            raise NotImplementedError('Multiplication of {} with {} not supported'.format(type(self), type(other)))
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = type(self)({other * key: val for key, val in self.items()})
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
        """Return an empty model object that inherits the size and momenta"""
        result = type(self)({})
        result.shape = self.shape
        result.momenta = self.momenta
        return result

    def transform_symbolic(self, func):
        """Transform keys by applying func to all of them. Useful for
        symbolic substitutions, differentiation, etc. If key is a BlochCoeff
        the substitution is only applied to the keys."""
        if self == {}:
            result = self.zeros_like()
        else:
            result = sum(type(self)({func(key): val}) for key, val in self.items())
            # Remove possible duplicate keys that only differ in constant factors
            result.shape = self.shape
            result.momenta = self.momenta
        return result

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R"""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)

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
        result = type(self)({key.subs(sympy.I, -sympy.I): val.conj()
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

    def tosympy(self, nsimplify=False):
        # Return sympy representation of the term
        # If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas
        if not nsimplify:
            return sympy.sympify(sum(key * val for key, val in self.data.items()))
        else:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            return sympy.MatAdd(*[key * sympy.Matrix(vnsimplify(val))
                              for key, val in self.data.items()]).doit()

    def copy(self):
        return deepcopy(self)
    
    def lambdify(self, nsimplify=False, *, onsite=False, hopping=False):
        """Return a callable object for the model, with sympy symbols as
        parameters.

        Parameters
        ----------
        nsimplify: bool, default False
            Whether or not to attempt to rewrite numerical coefficients as
            exact symbols in sympification.
        onsite : bool, default False
            If True, adds 'site' as the first argument to the callable object.
            Helpful for passing Model objects to kwant Builder objects as
            onsite functions.
        hopping : bool, default False
            If True, adds 'site1' and 'site2' as the first two arguments to
            the callable object.
            Helpful for passing Model objects to kwant Builder objects as
            hopping functions.
            
        Notes:
        onsite and hopping are mutually exclusive. If both are set to True,
        an error is thrown.
        """
        # Replace 'e' with the numerical value
        expr = self.tosympy(nsimplify=nsimplify).subs({'e': np.e})
        # Needed if expr is an array with 1 element, because .tosympy
        # returns a scalar then.
        try:
            expr = sympy.Matrix(expr).reshape(*expr.shape)
        except TypeError:
            pass
        args = sorted([s.name for s in expr.atoms(sympy.Symbol)])
        if onsite and not hopping:
            args = ['site'] + args
        elif hopping and not onsite:
            args = ['site1', 'site2'] + args
        elif hopping and onsite:
            raise ValueError("'hopping' and 'onsite' are mutually exclusive")
        return sympy.lambdify(args, expr)


class BlochModel(Model):
    def __init__(self, hamiltonian={}, locals=None, momenta=[0, 1, 2]):
        """Class to store Bloch Hamiltonian families.
        Can be used to efficiently store any matrix valued function.
        Implements many sympy and numpy methods. Arithmetic operators are overloaded,
        such that `*` corresponds to matrix multiplication.

        Parameters
        ----------
        hamiltonian : str, SymPy expression or dict
            Symbolic representation of a Hamiltonian.  It is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            {BlochCoeff: np.ndarray} with all arrays the same square size.
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
        if isinstance(hamiltonian, Model):
            # Recast keys into BlochCoeffs
            self.__init__({_to_bloch_coeff(key, hamiltonian.momenta): val
                           for key, val in hamiltonian.items()}, momenta=hamiltonian.momenta)
        elif isinstance(hamiltonian, abc.Mapping):
            keys = hamiltonian.keys()
            symbolic = all(isinstance(k, Basic) for k in keys)
            hopping = all(isinstance(k, BlochCoeff) for k in keys)
            if not (symbolic or hopping):
                raise ValueError('All keys must have the same type (sympy expression or BlochCoeff).')
            if hopping or hamiltonian == {}:
                # initialize as dict
                super(Model, self).__init__(hamiltonian)
                self.momenta = _find_momenta(momenta)
                self.shape = _find_shape(self.data)
            elif symbolic:
                self.__init__(Model(hamiltonian, locals, momenta))
        else:
            # Use Model to parse input
            self.__init__(Model(hamiltonian, locals, momenta))

    def transform_symbolic(self, func):
        raise NotImplementedError('`transform_symbolic` is not implemented for `BlochModel`')

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R"""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)
        # do rotation on hopping vectors with transpose matrix
        R_T = np.array(R).T
        return BlochModel({BlochCoeff(R_T @ hop, coeff): val
                      for (hop, coeff), val in self.items()}, momenta=momenta)

    def conj(self):
        """Complex conjugation"""
        result = BlochModel({BlochCoeff(-hop, coeff.subs(sympy.I, -sympy.I)): val.conj()
                            for (hop, coeff), val in self.items()})
        result.momenta = self.momenta
        return result

    def tosympy(self, nsimplify=False):
        # Return sympy representation of the term
        # If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas
        return self.tomodel(nsimplify=nsimplify).tosympy(nsimplify)

    def tomodel(self, nsimplify=False):
        return Model({key.tosympy(self.momenta, nsimplify=nsimplify): val
                      for key, val in self.items()}, momenta=self.momenta)


def _to_bloch_coeff(key, momenta):
    """Transform sympy expression to BlochCoeff is possible.
    `key` should be a single term with no sum and no power except for
    the Bloch factor."""
    # Key is an exponential
    if isinstance(key, sympy.power.Pow):
        expo = key
        coeff = sympy.numbers.One()
    # Key is the product of an exponential and some symbols.
    elif sympy.power.Pow in [type(arg) for arg in key.args]:
        find_expo = [ele for ele in key.args if type(ele) == sympy.power.Pow]
        assert len(find_expo) == 1
        expo = find_expo[0]
        coeff = sympy.simplify(key / expo)
    # Key contains no exponentials, then it is a constant
    else:
        hop = np.zeros((len(momenta,)))
        coeff = key
        expo = None
    # Extract hopping vector from exponential
    if expo is not None:
        args = expo.args
        assert args[0] == e
        assert type(args[1]) in (sympy.Mul, sympy.Add)  # The argument
        # Pick out the real space part, remove the complex i
        hop = np.array([args[1].coeff(momentum)/sympy.I
                          for momentum in momenta]).astype(float)
    bloch_coeff = BlochCoeff(hop, coeff)
    ### TODO: add tests to make sure it worked. This fails because the extra 1..
    # if not bloch_coeff.tosympy(momenta) == sympy.N(key):
    #     raise ValueError('Error transforming key {} to BlochCoeff {}.'.format(key, bloch_coeff.tosympy(momenta)))
    return bloch_coeff

def _find_momenta(momenta):
    if all([type(i) is int for i in momenta]):
        return [_commutative_momenta[i] for i in momenta]
    else:
        _momenta = [kwant_continuum.sympify(k) for k in momenta]
        return [kwant_continuum.make_commutative(k, k)
                        for k in _momenta]

def _find_shape(data):
    # Make sure every matrix has the same size
    if data == {}:
        return None
    else:
        shape = next(iter(data.values())).shape
        if not all([v.shape == shape for v in data.values()]):
            raise ValueError('All terms must have the same shape')
        return shape
