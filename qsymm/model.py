import numpy as np
import scipy
import tinyarray as ta
import scipy.linalg as la
import itertools as it
from copy import copy, deepcopy
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

    def __copy__(self):
        hop, coeff = self
        return BlochCoeff(copy(hop), copy(coeff))

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

    def __init__(self, hamiltonian=None, locals=None, momenta=(0, 1, 2), interesting_keys=None,
                 symbol_normalizer=None, restructure_dict=False):
        """
        General class to efficiently store any matrix valued function.
        The Model represents `sum(symbol * value)`, where `symbol` is a symbolic
        expression, and `value` can be scalar, array (both dense and sparse)
        or LinearOperator. The internal structure is a dict `{symbol: value}`.
        Implements many sympy and numpy methods and arithmetic operators.
        Multiplication is distributed over the sum, `*` is passed down to
        both symbols and values, `@` is passed to symbols as `*` and to values
        as `@`. By default symbols are sympified and assumed commutative.

        Parameters
        ----------
        hamiltonian : str, SymPy expression, dict or None (default)
            Symbolic representation of a Hamiltonian.  If a string, it is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            `{symbol: array}` with all arrays the same size (dense or sparse).
            `symbol` by default is passed through sympy.sympify, and should
            consist purely of a product of symbolic coefficients, no constant
            factors other than 1. `None` initializes a zero `Model`.
        locals : dict or ``None`` (default)
            Additional namespace entries for `~kwant_continuum.sympify`.  May be
            used to simplify input of matrices or modify input before proceeding
            further. For example:
            ``locals={'k': 'k_x + I * k_y'}`` or
            ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
        interesting_keys : iterable of expressions (optional)
            Set of symbolic coefficients that are kept, anything that does not
            appear here is discarded. Useful for perturbative calculations where
            only terms to a given order are needed. By default all keys are kept.
        momenta : iterable of int or list of Sympy objects
            Indices of momentum variables from ['k_x', 'k_y', 'k_z']
            or a list of names for the momentum variables as sympy symbols.
            Momenta are treated the same as other keys for the purpose of
            `interesting_keys`, need to list interesting powers of momenta.
        symbol_normalizer : callable (optional)
            Function to apply symbols when initializing with dict. By default the
            keys are passed through `sympy.sympify` and `sympy.expand_power_exp`.
        restructure_dict : bool, default False
            Whether to clean input dict by splitting summands in symbols,
            moving numerical factors in the symbols to values, removing entries
            with values np.allclose to zero
        """
        self.momenta = _find_momenta(momenta)

        if interesting_keys is not None:
            self.interesting_keys = {sympy.sympify(k) for k in interesting_keys}
        else:
            self.interesting_keys = set()

        if hamiltonian is None:
            hamiltonian = {}

        if hamiltonian == {} or isinstance(hamiltonian, abc.Mapping):
            if symbol_normalizer is None:
                symbol_normalizer = lambda x: sympy.expand_power_exp(sympy.sympify(x))
            # Initialize as dict sympifying the keys
            super().__init__({symbol_normalizer(k): v for k, v in hamiltonian.items()
                              if symbol_normalizer(k) in self.interesting_keys
                                 or not self.interesting_keys})
        else:
            # Try to parse the input with kwant_continuum.sympify
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
            restructure_dict = True

        # Keep track of whether this is a dense array
        self._isarray = any(isinstance(val, np.ndarray) for val in self.values())

        if restructure_dict:
            # Clean internal data by:
            # * splitting summands in keys
            # * moving numerical factors to values
            # * removing entries which values care np.allclose to zero
            new_data = defaultdict(lambda: list())
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
            if self.shape == ():
                #scalar
                return np.complex128(0)
            elif self._isarray:
                # Return dense zero array if dense
                return np.zeros(self.shape, dtype=complex)
            else:
                # Otherwise return a csr_matrix
                return scipy.sparse.csr_matrix(self.shape, dtype=complex)
        else:
            return None

    def __eq__(self, other):
        # Call allclose with default tolerances
        return self.allclose(other)

    def __add__(self, other):
        # Addition of Models. It is assumed that both Models are
        # structured correctly, every key is in standard form.
        # Define addition of 0 and {}
        if (not isinstance(other, type(self)) and (other == 0 or other == {})
            or (isinstance(other, type(self)) and other.data=={})):
            result = self.copy()
        elif isinstance(other, type(self)):
            # other is not empty, so the result is not empty
            if self.momenta != other.momenta:
                raise ValueError("Can only add Models with the same momenta")
            result = self.zeros_like()
            for key in self.keys() & other.keys():
                total = self[key] + other[key]
                # If only one is sparse matrix, the result is np.matrix, recast it to np.ndarray
                if isinstance(total, np.matrix):
                    total = total.A
                result[key] = total
            for key in self.keys() - other.keys():
                result[key] = copy(self[key])
            for key in other.keys() - self.keys():
                result[key] = copy(other[key])
            result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
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
        result = self.zeros_like()
        result.data = {key: -val for key, val in self.items()}
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        result = self.zeros_like()
        if isinstance(other, Number):
            result.data = {key: val * other for key, val in self.items()}
        elif isinstance(other, Basic):
            result.data = {key * other: val for key, val in self.items()
                           if (key * other in interesting_keys or not interesting_keys)}
        elif isinstance(other, Model):
            if self.momenta != other.momenta:
                raise ValueError("Can only multiply Models with the same momenta")
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum(type(self)({k1 * k2: v1 * v2}, interesting_keys=interesting_keys)
                          for (k1, v1), (k2, v2) in it.product(self.items(), other.items())
                          if (k1 * k2 in interesting_keys or not interesting_keys))
            # Find out the shape of the result even if it is empty
            if result is 0:
                result = self.zeros_like()
                result.shape = (self[1] * other[1]).shape
            else:
                result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
            result.momenta = self.momenta.copy()
        else:
            # Otherwise try to multiply every value with other
            result.data = {key: val * other for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (self[1] * other).shape
            result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = self.zeros_like()
            result.data = {other * key: copy(val) for key, val in self.items()}
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: other * val for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (other * self[1]).shape
            result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
        return result

    def __matmul__(self, other):
        # Multiplication by arrays and Model
        if isinstance(other, Model):
            if self.momenta != other.momenta:
                raise ValueError("Can only multiply Models with the same momenta")
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum(type(self)({k1 * k2: v1 @ v2}, interesting_keys=interesting_keys)
                          for (k1, v1), (k2, v2) in it.product(self.items(), other.items())
                          if (k1 * k2 in interesting_keys or not interesting_keys))
            # Find out the shape of the result even if it is empty
            if result is 0:
                result = self.zeros_like()
                result.shape = (self[1] * other[1]).shape
            else:
                result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
            result.momenta = self.momenta.copy()
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: val @ other for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (self[1] @ other).shape
            result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
        return result

    def __rmatmul__(self, other):
        # Left multiplication by arrays
        result = self.zeros_like()
        result.data = {key: other @ val for key, val in self.items()}
        result.shape = _find_shape(result.data) if result.data else (other @ self[1]).shape
        result._isarray = any(isinstance(val, np.ndarray) for val in result.values())
        return result

    def __truediv__(self, other):
        result = self.zeros_like()

        if isinstance(other, Number):
            result.data = {key : val * (1/other) for key, val in self.items()}
        else:
            raise TypeError(
                "unsupported operand type for /: {} and {}".format(type(self), type(other)))
        return result

    def __repr__(self):
        result = ['{']
        for k, v in self.items():
            result.extend([str(k), ':\n', str(v), ',\n\n'])
        result.append('}')
        return "".join(result)

    def zeros_like(self):
        """Return an empty model object that inherits the other properties"""
        result = type(self)()
        result.interesting_keys = self.interesting_keys.copy()
        result.momenta = self.momenta.copy()
        result.shape = self.shape
        result._isarray = self._isarray
        return result

    def transform_symbolic(self, func):
        """Transform keys by applying func to all of them. Useful for
        symbolic substitutions, differentiation, etc."""
        # Add possible duplicate keys that only differ in constant factors
        result = sum((type(self)({func(key): copy(val)},
                                 restructure_dict=True,
                                 momenta=self.momenta.copy())
                         for key, val in self.items()),
                     self.zeros_like())
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
        `sympy.Expr.subs()` for details.

        Allows for the replacement of momenta in the Model object.
        Replacing a momentum k with a sympy.Symbol object p replaces
        the momentum k with p in the Model.
        Replacing a momentum k with a number removes the momentum k
        from the Model momenta.
        Replacing a momentum k with a sympy expression that does not contain
        any of the Model.momenta also removes the momentum k from the
        momenta.
        """
        # Allowed inputs are an old, new pair, or
        # a list or dictionary of old, new pairs.
        # Bring them all to the form of a list of old, new pairs.
        if len(args) == 2:  # Input is a single pair
            args = ([(args[0], args[1])], )
        elif isinstance(args[0], dict): # Input is a dictionary
            args = ([(key, value) for key, value in args[0].items()], )

        momenta = self.momenta.copy()
        for (old, new) in args[0]:
            # Substitution of a momentum variable with a symbol
            # is a renaming of the momentum.
            if old in momenta and isinstance(new, sympy.Symbol):
                momenta = [momentum if old is not momentum else new
                           for momentum in momenta]
            # If no momenta appear in the replacement for a momentum, we consider
            # that momentum removed.
            # Replacement is not a sympy object.
            elif not isinstance(new, sympy.Basic):
                momenta = [momentum for momentum in momenta if old is not momentum]
            # Replacement is a sympy object, but does not contain momenta.
            elif not any([momentum in new.atoms() for momentum in momenta]):
                momenta = [momentum for momentum in momenta if old is not momentum]
        substituted = self.transform_symbolic(lambda x: x.subs(*args, **kwargs))
        substituted.momenta = momenta
        # If there are exponentials, evaluate any numerical exponents,
        # so they can be moved to the matrix valued part of the Model
        result = substituted.zeros_like()
        for key, value in substituted.items():
            # Expand sums in the exponent to products of exponentials,
            # find all exponentials.
            key = sympy.expand(key, power_base=True, power_exp=True,
                               mul=True, log=False, multinomial=False)
            find_expos = [ele for ele in key.args if ele.is_Pow]
            if len(find_expos):
                rest = key / np.prod(find_expos)
                # If an exponential evaluates to a number, replace it with that number.
                # Otherwise, leave the exponential unchanged.
                expos = [expo.subs(e, np.e).evalf() if expo.subs(e, np.e).evalf().is_number
                         else expo for expo in find_expos]
                result += type(substituted)({rest * np.prod(expos): value}, momenta=momenta, restructure_dict=True)
            else:
                result += type(substituted)({key: value}, momenta=momenta, restructure_dict=True)
        return result

    def conj(self):
        """Complex conjugation"""
        result = self.zeros_like()
        # conjugation is bijective, if self was properly formatted, so is this
        result.data = {key.subs(sympy.I, -sympy.I): val.conj()
                        for key, val in self.items()}
        return result

    def T(self):
        """Transpose"""
        result = self.zeros_like()
        result.data = {key: val.T for key, val in self.items()}
        result.shape = self.shape[::-1]
        return result

    def trace(self):
        result = self.zeros_like()
        result.data = {key: np.sum(val.diagonal()) for key, val in self.items()}
        result.shape = ()
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
            result = sympy.sympify(sum(key * val for key, val in self.toarray().items()))
        else:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            result = sympy.MatAdd(*[key * sympy.Matrix(vnsimplify(val))
                                    for key, val in self.toarray().items()]).doit()
        if any([isinstance(result, matrix_type) for matrix_type in (sympy.MatrixBase,
                                                                    sympy.ImmutableDenseMatrix,
                                                                    sympy.ImmutableDenseNDimArray)]):
            result = sympy.Matrix(result).reshape(*result.shape)
        return result

    def evalf(self, subs=None):
        return sum(float(key.evalf(subs=subs)) * val for key, val in self.items())

    def tocsr(self):
        result = self.zeros_like()
        for key, val in self.items():
            if isinstance(val, (Number, np.ndarray, scipy.sparse.spmatrix)):
                result[key] = scipy.sparse.csr_matrix(val, dtype=complex)
            else:
                # LinearOperator doesn't support multiplication with sparse matrix
                val = scipy.sparse.csr_matrix(val @ np.eye(val.shape[-1], dtype=complex), dtype=complex)
        result._isarray = False
        return result

    def toarray(self):
        result = self.zeros_like()
        for key, val in self.items():
            if isinstance(val, np.ndarray):
                result[key] = val
            elif isinstance(val, Number):
                result[key] = np.asarray(val)
            elif scipy.sparse.spmatrix:
                result[key] = val.A
            else:
                 val = val @ np.eye(val.shape[-1], dtype=complex)
        result._isarray = True
        return result

    def copy(self):
        result = self.zeros_like()
        # This is faster than deepcopy of the dict
        result.data = {copy(k): copy(v) for k, v in self.items()}
        return result

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

    def reshape(self, *args, **kwargs):
        result = self.zeros_like()
        result.data = {key: val.reshape(*args, **kwargs) for key, val in self.items()}
        result.shape = _find_shape(result.data)
        return result

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        # Test whether two arrays are approximately equal
        if other == {} or other == 0:
            if self.data == {}:
                return True
            else:
                return all(allclose(val, 0, rtol, atol, equal_nan) for val in self.values())
        else:
            return all(allclose(self[key], other[key], rtol, atol, equal_nan)
                       for key in self.keys() | other.keys())


class BlochModel(Model):
    def __init__(self, hamiltonian=None, locals=None, momenta=(0, 1, 2),
                 interesting_keys=None):
        """
        Class to efficiently store matrix valued Bloch Hamiltonians.
        The BlochModel represents `sum(BlochCoeff * value)`, where `BlochCoeff`
        is a symbolic representation of coefficient and periodic functions.
        `value` can be scalar, array (both dense and sparse)
        or LinearOperator. The internal structure is a dict `{BlochCoeff: value}`.
        Implements many sympy and numpy methods and arithmetic operators.
        Multiplication is distributed over the sum, `*` is passed down to
        both symbols and values, `@` is passed to symbols as `*` and to values
        as `@`. By default symbols are sympified and assumed commutative.

        Parameters
        ----------
        hamiltonian : Model, str, SymPy expression, dict or None (default)
            Symbolic representation of a Hamiltonian.  If a string, it is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            `{symbol: array}` with all arrays the same size (dense or sparse).
            If symbol is not a BlochCoeff, it is passed through sympy.sympify,
            and should consist purely of a product of symbolic coefficients,
            no constant factors other than 1. `symbol` is then converted to BlochCoeff.
            `None` initializes a zero `BlochModel`.
        locals : dict or ``None`` (default)
            Additional namespace entries for `~kwant_continuum.sympify`.  May be
            used to simplify input of matrices or modify input before proceeding
            further. For example:
            ``locals={'k': 'k_x + I * k_y'}`` or
            ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
        momenta : iterable of int or list of Sympy objects
            Indices of momenta the monomials depend on from 'k_x', 'k_y' and 'k_z'
            or a list of names for the momentum variables. Ignored when
            initialized with Model.
        interesting_keys : iterable of BlochCoeff (optional)
            Set of symbolic coefficients that are kept, anything that does not
            appear here is discarded. Useful for perturbative calculations where
            only terms to a given order are needed. By default all keys are kept.
            Ignored when initialized with Model.
        """
        if hamiltonian is None:
            hamiltonian = {}
        if isinstance(hamiltonian, Model):
            # First initialize an empty BlochModel, this is the same as init for Model
            self.__init__(hamiltonian={},
                          locals=locals,
                          momenta=hamiltonian.momenta,
                          interesting_keys=hamiltonian.interesting_keys)
            # Initialize same as input model, so __missing__ works
            self._isarray = hamiltonian._isarray
            self.shape = hamiltonian.shape
            # Recast keys into BlochCoeffs, if some keys are different but close,
            # such that BlochCoeff is the same, collect them.
            for key, val in hamiltonian.items():
                self[_to_bloch_coeff(key, hamiltonian.momenta)] += val
        elif isinstance(hamiltonian, abc.Mapping):
            keys = hamiltonian.keys()
            symbolic = all(not isinstance(k, BlochCoeff) for k in keys)
            hopping = all(isinstance(k, BlochCoeff) for k in keys)
            if not (symbolic or hopping):
                raise ValueError('All keys must have the same type (sympy expression or BlochCoeff).')
            if hopping or hamiltonian == {}:
                # initialize as Model without any of the preprocessing
                super().__init__(hamiltonian,
                                 locals=locals,
                                 momenta=momenta,
                                 interesting_keys=interesting_keys,
                                 symbol_normalizer=lambda x: x,
                                 restructure_dict=False)
            elif symbolic:
                # First cast it to model, then try to interpret it as BlochModel
                self.__init__(Model(hamiltonian,
                                    locals=locals,
                                    momenta=momenta,
                                    interesting_keys=interesting_keys))
        else:
            # Use Model to parse input
            self.__init__(Model(hamiltonian,
                                locals=locals,
                                momenta=momenta,
                                interesting_keys=interesting_keys))

    def transform_symbolic(self, func):
        raise NotImplementedError('`transform_symbolic` is not implemented for `BlochModel`')

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R"""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)
        # do rotation on hopping vectors with transpose matrix
        R_T = np.array(R).astype(float).T
        return BlochModel({BlochCoeff(R_T @ hop, coeff): val
                      for (hop, coeff), val in self.items()}, momenta=momenta)

    def conj(self):
        """Complex conjugation"""
        result = self.zeros_like()
        result.data = {BlochCoeff(-hop, coeff.subs(sympy.I, -sympy.I)): val.conj()
                            for (hop, coeff), val in self.items()}
        return result

    def subs(self, *args, **kwargs):
        model = self.tomodel(nsimplify=False)
        result = model.subs(*args, **kwargs)
        return BlochModel(result, momenta=self.momenta)

    def tosympy(self, nsimplify=False):
        # Return sympy representation of the term
        # If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas
        return self.tomodel(nsimplify=nsimplify).tosympy(nsimplify)

    def tomodel(self, nsimplify=False):
        return Model({key.tosympy(self.momenta, nsimplify=nsimplify): val
                      for key, val in self.items()}, momenta=self.momenta)


def _to_bloch_coeff(key, momenta):
    """Transform sympy expression to BlochCoeff if possible."""

    def is_hopping_expo(expo):
        # Check whether a sympy exponential represents a hopping.
        base, exponent = expo.as_base_exp()
        if base == e and any([momentum in exponent.atoms()
                              for momentum in momenta]):
            return True
        else:
            return False

    # We combine exponentials with the same base and exponent.
    key = sympy.powsimp(key, combine='exp')
    # Expand multiplication of brackets into sums.
    key = sympy.expand(key, power_base=False, power_exp=False,
                       mul=True, log=False, multinomial=False)
    if isinstance(key, sympy.add.Add):
        raise ValueError("Key cannot be a sum of terms.")
    # Key is a single exponential.
    if isinstance(key, sympy.power.Pow):
        base, exp = key.as_base_exp()
        # If the exponential is a hopping, store it
        # with coefficient 1.
        if is_hopping_expo(key):
            hop_expo = key
            coeff = sympy.numbers.One()
        # If it is not a hopping, it belongs to the coeff.
        else:
            hop, coeff, hop_expo = np.zeros((len(momenta,))), key, None
    # Key is the product of an exponential and some extra stuff.
    elif sympy.power.Pow in [type(arg) for arg in key.args]:
        # Check that a natural exponential is present, which also
        # includes momenta in its arguments.
        # First find all exponentials.
        find_expos = [ele for ele in key.args if ele.is_Pow]
        # Then pick out exponentials that are hoppings.
        hop_expos = [expo for expo in find_expos if is_hopping_expo(expo)]
        # We should find at most one exponential that represents a
        # hopping, because all exponentials with the same base have been
        # combined.
        if len(hop_expos) == 1:
            hop_expo = hop_expos[0]
            coeff = sympy.simplify(key / hop_expo)
        # If none of the exponentials match the hopping structure, the
        # exponentials that are present are parts of the coefficient,
        # so this is an onsite term.
        elif not len(hop_expos):
            hop, coeff, hop_expo = np.zeros((len(momenta,))), key, None
        # Should never be called.
        else:
            raise ValueError("Unable to read the hoppings in "
                             "conversion to BlochCoeff.")
    # If the key contains no exponentials, then it is not a hopping.
    else:
        hop, coeff, hop_expo = np.zeros((len(momenta,))), key, None
    # Extract hopping vector from exponential
    # If the exponential contains more arguments than the hopping,
    # append it to coeff.
    if hop_expo is not None:
        base, exponent = hop_expo.as_base_exp()
        assert base == e
        assert type(exponent) in (sympy.Mul, sympy.Add)
        # Pick out the real space part, remove the complex i,
        # expand any brackets if present.
        arg = exponent.expand()
        # Check that the momenta all have i as a prefactor
        momenta_present = [momentum for momentum in momenta
                           if momentum in arg.atoms()]
        assert all([sympy.I in (arg.coeff(momentum)).atoms()
                    for momentum in momenta_present]), \
               "Momenta in hopping exponentials should have a complex prefactor."
        hop = [sympy.expand(arg.coeff(momentum)/sympy.I)
               for momentum in momenta]
        # We do not allow sympy symbols in the hopping, should
        # be numerical values only.
        assert not any([isinstance(item, sympy.symbol.Symbol)
                        for ele in hop for item in ele.atoms()
                        if isinstance(ele, sympy.Expr)]), \
                        "Real space part of the hopping " \
                        "must be numbers, not symbols."
        # If the exponential contains something extra other than the
        # hopping part, we append it to the coefficient.
        spatial_arg = sympy.I*sum([ele*momentum for ele, momentum in zip(momenta, hop)])
        diff = sympy.nsimplify(sympy.expand(arg - spatial_arg))
        coeff = sympy.simplify(coeff * e**diff)
        hop = np.array(hop).astype(float)
    # Make sure there is no momentum dependence in the coefficient.
    assert not any([momentum in coeff.atoms() for momentum in momenta]), \
                "All momentum dependence should be confined to " \
                "hopping exponentials."
    bloch_coeff = BlochCoeff(hop, coeff)
    # Transform back, compare to make sure everything is consistent.
    # Tricky to compare sympy objects...
#     if not (sympy.simplify(bloch_coeff.tosympy(momenta, nsimplify=True)) ==
#             key):
#         raise ValueError('Error transforming key {} to BlochCoeff {}.'.format(key, bloch_coeff.tosympy(momenta)))
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
        val = next(iter(data.values()))
        if isinstance(val, Number):
            shape = ()
            if not all(isinstance(v, Number) for v in data.values()):
                raise ValueError('All terms must have the same shape')
            # Recast numbers to numpy complex128
            for key, val in data.items():
                data[key] = np.complex128(val)
        else:
            shape = val.shape
            if not all(v.shape == shape for v in data.values()):
                raise ValueError('All terms must have the same shape')
        return shape
