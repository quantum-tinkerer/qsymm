import numpy as np
import scipy
import tinyarray as ta
import scipy.linalg as la
from itertools import product
import copy as copy_module
from numbers import Number
from warnings import warn
from functools import lru_cache
import sympy
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef
from collections import defaultdict, abc, UserDict
from .linalg import prop_to_id, allclose

from . import kwant_continuum, _scipy_patch

_commutative_momenta = [kwant_continuum.make_commutative(k, k)
           for k in kwant_continuum.momentum_operators]

e = kwant_continuum.sympify('e')
I = kwant_continuum.sympify('I')


# Scipy sparse matrices defined a 'copy' method (which does
# a deep-copy of their data) but no '__copy__' method, to work
# correctly with the 'copy' module. We therefore make our own
# wrapper here that does the right thing
def copy(a):
    if callable(getattr(a, 'copy', None)):
        return a.copy()
    else:
        return copy_module.copy(a)


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
    """
    Container for Bloch coefficient in ``BlochModel``, in the form of
    ``(hop, coeff)``, equivalent to ``coeff * exp(I * hop.dot(k))``.
    """

    def __new__(cls, hop, coeff):
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
        if other == 1:
            return allclose(hop1, 0) and coeff1 == 1
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

    def __copy__(self):
        return self.copy()

    def copy(self):
        hop, coeff = self
        # Do not copy 'coeff', as Sympy objects are immutable anyway,
        # and making a copy breaks equality checking and hashing.
        return BlochCoeff(copy(hop), coeff)

    def tosympy(self, momenta, nsimplify=False):
        hop, coeff = self
        if nsimplify:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            hop = vnsimplify(hop)
        return coeff * e**(sum(I * ki * di for ki, di in zip(momenta, hop)))


class Model(UserDict):
    """
    Symbolic matrix-valued function that depends on momenta and other parameters.

    Implements the algebra of matrix valued functions.
    Implements many sympy and numpy methods and overrides arithmetic operators.
    Internally it represents ``sum(symbol * value)``, where ``symbol`` is a symbolic
    expression, and ``value`` can be scalar, array (both dense and sparse)
    or LinearOperator. This is accessible as a dict ``{symbol: value}``.

    Parameters
    ----------
    hamiltonian : str, SymPy expression, dict or None (default)
        Symbolic representation of a Hamiltonian.  If a string, it is
        first converted to a SymPy expression using `kwant_continuum.sympify`.
        If a dict is provided, it should have the form
        ``{symbol: array}`` with all arrays the same size (dense or sparse).
        ``symbol`` by default is passed through sympy.sympify, and should
        consist purely of a product of symbolic coefficients, no constant
        factors other than 1, except if ``normalize=True``. ``None`` initializes
        a zero ``Model``.
    locals : dict or ``None`` (default)
        Additional namespace entries for `~kwant_continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
    keep : iterable of expressions (optional)
        Set of symbolic coefficients that are kept, anything that does not
        appear here is discarded. Useful for perturbative calculations where
        only terms to a given order are needed. By default all keys are kept.
    momenta : iterable of strings or Sympy symbols
        Names of momentum variables, default ``('k_x', 'k_y', 'k_z')`` or
        corresponding sympy symbols. Momenta are treated the same as other
        keys for the purpose of `keep`.
    symbol_normalizer : callable (optional)
        Function applied to symbols when initializing the internal dict. By default the
        keys are passed through ``sympy.sympify`` and ``sympy.expand_power_exp``.
        Keys when accessing a term and keys in ``keep`` are also passed through
        ``symbol_normalizer``.
    normalize : bool, default False
        Whether to clean input dict by splitting summands in symbols,
        moving numerical factors in the symbols to values, removing entries
        with values allclose to zero. Ignored if hamiltonian is not a dict.
    shape : tuple or None (default)
        Shape of the Model, must match the shape of all the values. If not
        provided, it is automatically found based on the shape of the input.
        Must be provided if ``hamiltonian`` is ``None`` or ``{}``. Empty tuple
        corresponds to scalar values.
    format : class or None (default)
        Type of the values in the model. Supported types are
        ``np.complex128``, ``scipy.sparse.linalg.LinearOperator``, ``np.ndarray``,
        and subclasses of ``scipy.sparse.spmatrix`` . If ``hamiltonian`` is
        provided as a dict, all values must be of this type, except for
        scalar values, which are recast to ``np.complex128``. If ``format`` is
        not provided, it is inferred from the type of the values. Must be
        provided if ``hamiltonian`` is `None` or ``{}``. If ``hamiltonian`` is
        not a dictionary, ``format`` is ignored an set to ``np.ndarray``.

    Notes
    -----
    Sympy symbols are immutable and references to the same symbols is
    stored in different Models. Be warned that setting any assumptions
    for symbols (such as ``real``) will result in an identically named,
    but different symbol, and these are not handled properly. Model assumes
    that all sympy symbols are real without any assumptions explicitly set.
    """

    # Make it work with numpy arrays
    __array_ufunc__ = None

    def __init__(
        self,
        hamiltonian=None,
        locals=None,
        momenta=('k_x', 'k_y', 'k_z'),
        keep=None,
        symbol_normalizer=None, normalize=False, shape=None, format=None
    ):
        if hamiltonian is None:
            hamiltonian = {}
        if symbol_normalizer is None:
            symbol_normalizer = _symbol_normalizer
        self.momenta = _find_momenta(tuple(momenta))

        if keep is not None:
            self.keep = {symbol_normalizer(k) for k in keep}
        else:
            self.keep = set()

        if hamiltonian == {} or isinstance(hamiltonian, abc.Mapping):
            # Initialize as dict sympifying the keys
            self.data = {symbol_normalizer(k): v for k, v in hamiltonian.items()
                              if symbol_normalizer(k) in self.keep
                                 or not self.keep}

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
            normalize = True

        # Find shape and format
        self.shape = shape
        self.format = format
        if self.shape is None or self.format is None:
            if self.data == {}:
                # raise ValueError('Must provide `shape` and `format` when initializing empty Model.')
                warn('Provide `shape` and `format` when initializing empty Model.', DeprecationWarning)
            else:
                val = next(iter(self.values()))
                shape, format = _shape_and_format(val)
                self.shape = (shape if self.shape is not None else shape)
                self.format = (format if self.format is not None else format)
        if shape == ():
            # Recast numbers to np.complex128
            self.data = {k: np.complex128(v) for k, v in self.items()}
        if not all(issubclass(type(v), format) for v in self.values()):
            raise ValueError('All values must have the same `format`.')
        if not all(v.shape == shape for v in self.values()):
            raise ValueError('All values must have the same `shape`.')

        if normalize:
            # Clean internal data by:
            # * splitting summands in keys
            # * moving numerical factors to values
            # * removing entries which values care np.allclose to zero
            # Do not copy key, as Sympy objects are immutable anyway,
            # and making a copy breaks equality checking and hashing.
            old_data = {key: copy(val) for key, val in self.items()}
            self.data = {}
            for key, val in old_data.items():
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
                    self[new_key] += new_val
            # remove zero entries, apply symbol_normalizer
            self.data = {symbol_normalizer(k): v for k, v in self.items() if not allclose(v, 0)}

    # Make sure values have the correct format
    def __setitem__(self, key, item):
        if (isinstance(item, self.format) and self.shape == item.shape):
            self.data[key] = item
        elif (isinstance(item, Number) and self.shape == ()):
            self.data[key] = np.complex128(item)
        else:
            raise ValueError('Format of item ({}) must match the format ({}) '
                             'and shape ({}) of Model'.format(item, self.format, self.shape))

    # Allow getting values using text keys
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif _symbol_normalizer(key) in self.data:
            return self.data[_symbol_normalizer(key)]
        else:
            return self.__missing__(key)

    # Defaultdict functionality
    def __missing__(self, key):
        if self.format is np.complex128:
            #scalar
            return np.complex128(0)
        elif self.format is np.ndarray:
            # Return dense zero array if dense
            return np.zeros(self.shape, dtype=complex)
        elif issubclass(self.format, scipy.sparse.spmatrix):
            # Return a zero sparse matrix of the same type
            return self.format(self.shape, dtype=complex)
        elif issubclass(self.format, scipy.sparse.linalg.LinearOperator):
            return scipy.sparse.linalg.aslinearoperator(
                scipy.sparse.csr_matrix(self.shape, dtype=complex))

    def __eq__(self, other):
        # Call allclose with default tolerances
        return self.allclose(other)

    def __add__(self, other):
        # Addition of Models. It is assumed that both Models are
        # structured correctly, every key is in standard form.

        # Useful for sum to work.
        if isinstance(other, Number) and other == 0:
            result = self.copy()
        # Temporarily allow adding malshaped empty Models
        elif (isinstance(other, type(self)) and other.data=={}):
            result = self.copy()
        elif (isinstance(other, type(self)) and self.data=={}):
            result = other.copy()
        elif isinstance(other, type(self)):
            if not (self.format is other.format and self.shape == other.shape):
                raise ValueError('Addition is only possible for Models with the same shape and data type.')
            # other is not empty, so the result is not empty
            if self.momenta != other.momenta:
                raise ValueError("Can only add Models with the same momenta")
            result = self.zeros_like()
            for key in self.keys() & other.keys():
                result[key] = self[key] + other[key]
            for key in self.keys() - other.keys():
                result[key] = copy(self[key])
            for key in other.keys() - self.keys():
                result[key] = copy(other[key])
        elif ((isinstance(other, self.format) and self.shape == other.shape)
              or (isinstance(other, Number) and self.shape == ())):
            # Addition of constants
            result = self.copy()
            result[1] += other
        else:
            raise NotImplementedError('Addition of {} with shape {} with {} not supported'.format(type(self), self.shape, type(other)))
        return result

    def __radd__(self, other):
        # Addition of monomials with other types.

        # Useful for sum to work.
        if isinstance(other, Number) and other == 0:
            result = self.copy()
        elif ((isinstance(other, self.format) and self.shape == other.shape)
              or (isinstance(other, Number) and self.shape == ())):
            # Addition of constants
            result = self.copy()
            result[1] += other
        else:
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))
        return result

    def __neg__(self):
        result = self.zeros_like()
        result.data = {key: -val for key, val in self.items()}
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        if isinstance(other, Number):
            result = self.zeros_like()
            result.data = {key: val * other for key, val in self.items()}
        elif isinstance(other, Basic):
            keep = self.keep
            result = sum((type(self)({key * other: copy(val)},
                                     keep=keep,
                                     momenta=self.momenta)
                         for key, val in self.items()
                         if (key * other in keep or not keep)),
                         self.zeros_like())
        elif isinstance(other, Model):
            if not (issubclass(self.format, (Number, np.ndarray)) or
                    issubclass(other.format, (Number, np.ndarray))):
                raise ValueError('Elementwise multiplication only allowed for scalar '
                                 'and ndarra data types. With sparse matrices use `@` '
                                 'for matrix multiplication.')
            if self.momenta != other.momenta:
                raise ValueError("Can only multiply Models with the same momenta")
            keep = self.keep | other.keep
            result = sum(type(self)({k1 * k2: v1 * v2},
                                    keep=keep,
                                    momenta=self.momenta)
                          for (k1, v1), (k2, v2) in product(self.items(), other.items())
                          if (k1 * k2 in keep or not keep))
            # Find out the shape of the result even if it is empty
            if isinstance(result, Number) and result == 0:
                result = self.zeros_like()
                result.shape, result.format = _shape_and_format(self[1] * other[1])
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: val * other for key, val in self.items()}
            result.shape, result.format = _shape_and_format(self[1] * other)
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            keep = self.keep
            # The order 'key * other' is important: we want to force
            # the implementation of __mul__ of 'key' to be used. This
            # is correct as long as the symbols in 'key' and 'other' commute.
            result = sum((type(self)({key * other: copy(val)},
                                     keep=keep,
                                     momenta=self.momenta)
                         for key, val in self.items()
                         if (key * other in keep or not keep)),
                         self.zeros_like())
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: other * val for key, val in self.items()}
            result.shape, result.format = _shape_and_format(other * self[1])
        return result

    def __matmul__(self, other):
        # Multiplication by arrays and Model
        if isinstance(other, Model):
            if self.momenta != other.momenta:
                raise ValueError("Can only multiply Models with the same momenta")
            keep = self.keep | other.keep
            result = sum(type(self)({k1 * k2: v1 @ v2},
                                    keep=keep,
                                    momenta = self.momenta)
                          for (k1, v1), (k2, v2) in product(self.items(), other.items())
                          if (k1 * k2 in keep or not keep))
            # Find out the shape of the result even if it is empty
            if isinstance(result, Number) and result == 0:
                result = self.zeros_like()
                result.shape, result.format = _shape_and_format(self[1] @ other[1])
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: val @ other for key, val in self.items()}
            result.shape, result.format = _shape_and_format(self[1] @ other)
        return result

    def __rmatmul__(self, other):
        # Left multiplication by arrays
        result = self.zeros_like()
        result.data = {key: other @ val for key, val in self.items()}
        result.shape, result.format = _shape_and_format(other @ self[1])
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

    def __copy__(self):
        return self.copy()

    def zeros_like(self):
        """Return an empty model object that inherits the other properties"""
        result = type(self)(shape=self.shape,
                            format=self.format)
        result.keep=self.keep.copy()
        result.momenta=self.momenta
        return result

    def transform_symbolic(self, func):
        """Transform keys by applying func to all of them. Useful for
        symbolic substitutions, differentiation, etc."""
        # Add possible duplicate keys that only differ in constant factors
        result = sum((type(self)({func(key): copy(val)},
                                 normalize=True,
                                 momenta=self.momenta)
                         for key, val in self.items()),
                     self.zeros_like())
        return result

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R."""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)

        k_prime = R @ sympy.Matrix(momenta)
        rotated_subs = {k: k_prime for k, k_prime in zip(momenta, k_prime)}

        def trf(key):
            return key.subs(rotated_subs, simultaneous=True)

        return self.transform_symbolic(trf)

    def subs(self, *args, **kwargs):
        """Substitute symbolic expressions. See documentation of
        ``sympy.Expr.subs()`` for details.

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

        momenta = self.momenta
        for (old, new) in args[0]:
            # Substitution of a momentum variable with a symbol
            # is a renaming of the momentum.
            if old in momenta and isinstance(new, sympy.Symbol):
                momenta = tuple(momentum if old is not momentum else new
                                for momentum in momenta)
            # If no momenta appear in the replacement for a momentum, we consider
            # that momentum removed.
            # Replacement is not a sympy object.
            elif not isinstance(new, sympy.Basic):
                momenta = tuple(momentum for momentum in momenta if old is not momentum)
            # Replacement is a sympy object, but does not contain momenta.
            elif not any([momentum in new.atoms() for momentum in momenta]):
                momenta = tuple(momentum for momentum in momenta if old is not momentum)
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
                result += type(substituted)({rest * np.prod(expos): value}, momenta=momenta, normalize=True)
            else:
                result += type(substituted)({key: value}, momenta=momenta, normalize=True)
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
        """Take trace of the matrix and return a scalar valued Model."""
        result = self.zeros_like()
        result.data = {key: np.sum(val.diagonal()) for key, val in self.items()}
        result.shape, result.format = (), np.complex128
        return result

    def value_list(self, key_list):
        """Return a list of the matrix coefficients corresponding to the keys in key_list."""
        return [self[key] for key in key_list]

    def around(self, decimals=3):
        """Return Model with matrices rounded to given number of decimals."""
        result = self.zeros_like()
        for key, val in self.items():
            val = np.around(val, decimals)
            if not np.allclose(val, 0):
                result[key] = val
        return result

    def tosympy(self, nsimplify=False):
        """Return sympy representation of the Model.
        If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas."""
        if not nsimplify:
            result = sympy.sympify(sum(key * val for key, val in self.toarray().items()))
        else:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            result = sympy.MatAdd(*[key * sympy.Matrix(vnsimplify(val))
                                    for key, val in self.toarray().items()]).doit()
        if isinstance(result, (sympy.MatrixBase,
                               sympy.ImmutableDenseMatrix,
                               sympy.ImmutableDenseNDimArray)):
            result = sympy.Matrix(result).reshape(*result.shape)
        return result

    def evalf(self, subs=None):
        """Evaluate using parameter values in `subs`."""
        return sum(float(key.evalf(subs=subs)) * val for key, val in self.items())

    def tocsr(self):
        """Convert to sparse csr format."""
        result = self.zeros_like()
        result.format = scipy.sparse.csr_matrix
        for key, val in self.items():
            if isinstance(val, (Number, np.ndarray, scipy.sparse.spmatrix)):
                result[key] = scipy.sparse.csr_matrix(val, dtype=complex)
            else:
                # LinearOperator doesn't support multiplication with sparse matrix
                val = scipy.sparse.csr_matrix(val @ np.eye(val.shape[-1], dtype=complex), dtype=complex)
        return result

    def toarray(self):
        """Convert to dense numpy ndarray format."""
        result = self.zeros_like()
        result.format = np.ndarray
        for key, val in self.items():
            if isinstance(val, np.ndarray):
                result[key] = val
            elif isinstance(val, Number):
                result[key] = np.asarray(val)
            elif scipy.sparse.spmatrix:
                result[key] = val.A
            else:
                 val = val @ np.eye(val.shape[-1], dtype=complex)
        return result

    def copy(self):
        """Return a copy."""
        result = self.zeros_like()
        # This is faster than deepcopy of the dict
        # Do not copy the keys, as Sympy objects (and BlochCoeffs) are
        # immutable anyway, and making a copy breaks equality checking and hashing.
        result.data = {k: copy(v) for k, v in self.items()}
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
        """Reshape, see numpy.reshape."""
        result = self.zeros_like()
        result.data = {key: val.reshape(*args, **kwargs) for key, val in self.items()}
        result.shape, result.format = _shape_and_format(self[1].reshape(*args, **kwargs))
        return result

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """Test whether two Models are approximately equal"""
        if other == {} or other == 0:
            if self.data == {}:
                return True
            else:
                return all(allclose(val, 0, rtol, atol, equal_nan) for val in self.values())
        else:
            return all(allclose(self[key], other[key], rtol, atol, equal_nan)
                       for key in self.keys() | other.keys())

    def eliminate_zeros(self, rtol=1e-05, atol=1e-08):
        """Return a model with small terms removed. Tolerances are
        in Frobenius matrix norm, relative tolerance compares to the
        value with largest norm."""
        if not issubclass(self.format, (np.ndarray, scipy.sparse.spmatrix)):
            raise ValueError('Operation only supported for Models with '
                             '`np.ndarray` or `scipy.sparse.spmatrix` data type.')
        # Write it explicitely so it works with sparse arrays
        norm = lambda mat: np.sqrt(np.sum(np.abs(mat)**2))
        max_norm = np.max([norm(val) for val in self.values()])
        tol = max(atol, max_norm * rtol)
        result = self.zeros_like()
        result.data = {key: copy(val) for key, val in self.items() if not norm(val) < tol}
        return result


class BlochModel(Model):
    """
    A ``Model`` where coefficients are periodic functions of momenta.

    Internally it is a ``sum(BlochCoeff * value)``, where ``BlochCoeff`` is
    a symbolic representation of coefficients and a periodic function of ``k``.
    ``value`` can be scalar, array (both dense and sparse) or LinearOperator.
    This is accessible as a dict ``{BlochCoeff: value}``.

    Parameters
    ----------
    hamiltonian : Model, str, SymPy expression, dict or None (default)
        Symbolic representation of a Hamiltonian.  If a string, it is
        converted to a SymPy expression using ``kwant_continuum.sympify``.
        If a dict is provided, it should have the form
        ``{symbol: array}`` with all arrays the same size (dense or sparse).
        If symbol is not a BlochCoeff, it is passed through sympy.sympify,
        and should consist purely of a product of symbolic coefficients,
        no constant factors other than 1. `symbol` is then converted to BlochCoeff.
        `None` initializes a zero ``BlochModel``.
    locals : dict or ``None`` (default)
        Additional namespace entries for `~kwant_continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
    momenta : iterable of strings or Sympy symbols
        Names of momentum variables, default ``('k_x', 'k_y', 'k_z')`` or
        corresponding sympy symbols. Momenta are treated the same as other
        keys for the purpose of `keep`. Ignored when initialized with Model.
    keep : iterable of BlochCoeff (optional)
        Set of symbolic coefficients that are kept, anything that does not
        appear here is discarded. Useful for perturbative calculations where
        only terms to a given order are needed. By default all keys are kept.
        Ignored when initialized with Model.
    shape : tuple or None (default)
        Shape of the Model, must match the shape of all the values. If not
        provided, it is automatically found based on the shape of the input.
        Must be provided is ``hamiltonian`` is `None` or ``{}``. Empty tuple
        corresponds to scalar values. Ignored when initialized with Model.
    format : class or None (default)
        Type of the values in the model. Supported types are `np.complex128`,
        ``np.ndarray``, ``scipy.sparse.spmatrix`` and ``scipy.sparse.linalg.LinearOperator``.
        If ``hamiltonian`` is provided as a dict, all values must be of this type,
        except for scalar values, which are recast to ``np.complex128``.
        If ``format`` is not provided, it is inferred from the type of the values.
        If ``hamiltonian`` is not a dictionary, ``format`` is ignored and set to
        ``np.ndarray`` or ``hamiltonian.format`` if it is a ``Model``.
    """
    def __init__(self, hamiltonian=None, locals=None, momenta=('k_x', 'k_y', 'k_z'),
                 keep=None, shape=None, format=None):
        momenta = tuple(momenta)
        if hamiltonian is None:
            hamiltonian = {}
        if isinstance(hamiltonian, Model):
            # Use Model's init, only need to recast keys to BlochCoeff
            super().__init__(hamiltonian=hamiltonian.data,
                             locals=locals,
                             momenta=hamiltonian.momenta,
                             keep=hamiltonian.keep,
                             symbol_normalizer=lambda key: _bloch_normalizer(key, hamiltonian.momenta),
                             shape=hamiltonian.shape,
                             format=hamiltonian.format)
            # set these in case it was and empty Model
            self.format = hamiltonian.format
            self.shape = hamiltonian.shape
        elif isinstance(hamiltonian, abc.Mapping):
            keys = hamiltonian.keys()
            symbolic = all(not isinstance(k, BlochCoeff) for k in keys)
            hopping = all(isinstance(k, BlochCoeff) for k in keys)
            if hopping or hamiltonian == {}:
                # initialize as Model without any of the preprocessing
                super().__init__(hamiltonian,
                                 locals=locals,
                                 momenta=momenta,
                                 keep=keep,
                                 symbol_normalizer=lambda x: x,
                                 normalize=False,
                                 shape=shape,
                                 format=format,
                                )
            elif symbolic:
                # First cast it to model with restructuring, then try to interpret it as BlochModel
                self.__init__(Model(hamiltonian,
                                    locals=locals,
                                    momenta=momenta,
                                    keep=keep,
                                    normalize=True,
                                    shape=shape,
                                    format=format))
            else:
                raise ValueError('All keys must have the same type (sympy expression or BlochCoeff).')
        else:
            # Use Model to parse input
            self.__init__(Model(hamiltonian,
                                locals=locals,
                                momenta=momenta,
                                keep=keep,
                                shape=shape,
                                format=format))

    # Allow getting values using text keys
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif _bloch_normalizer(key, self.momenta) in self.data:
            return self.data[_bloch_normalizer(key, self.momenta)]
        else:
            return self.__missing__(key)

    def transform_symbolic(self, func):
        raise NotImplementedError('`transform_symbolic` is not implemented for `BlochModel`')

    def rotate_momenta(self, R):
        """Rotate momenta with rotation matrix R."""
        momenta = self.momenta
        assert len(momenta) == R.shape[0], (momenta, R)
        # do rotation on hopping vectors with transpose matrix
        R_T = np.array(R).astype(float).T
        result = self.zeros_like()
        result.data = {BlochCoeff(R_T @ hop, coeff): copy(val) for (hop, coeff), val in self.items()}
        return result

    def conj(self):
        """Complex conjugation."""
        result = self.zeros_like()
        result.data = {BlochCoeff(-hop, coeff.subs(sympy.I, -sympy.I)): val.conj()
                            for (hop, coeff), val in self.items()}
        return result

    def subs(self, *args, **kwargs):
        """Substitute symbolic expressions. See `Model.subs`."""
        model = self.tomodel(nsimplify=False)
        result = model.subs(*args, **kwargs)
        return BlochModel(result)

    def tosympy(self, nsimplify=False):
        """Return sympy representation of the Model.
        If nsimplify=True, attempt to rewrite numerical coefficients as exact formulas."""
        return self.tomodel(nsimplify=nsimplify).tosympy(nsimplify)

    def tomodel(self, nsimplify=False):
        """Convert to Model."""
        return Model({key.tosympy(self.momenta, nsimplify=nsimplify): copy(val)
                      for key, val in self.items()},
                     momenta=self.momenta,
                     keep={key.tosympy(self.momenta, nsimplify=nsimplify)
                                       for key in self.keep},
                     shape=self.shape,
                     format=self.format)


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
        if base != e or type(exponent) not in (sympy.Mul, sympy.Add):
            raise ValueError('Incorrect format of exponential.')
        # Pick out the real space part, remove the complex i,
        # expand any brackets if present.
        arg = exponent.expand()
        # Check that the momenta all have i as a prefactor
        momenta_present = [momentum for momentum in momenta
                           if momentum in arg.atoms()]
        if not all(
            [sympy.I in (arg.coeff(momentum)).atoms()
             for momentum in momenta_present]
        ):
            raise ValueError(
                "Momenta in hopping exponentials should have a complex prefactor."
            )
        hop = [sympy.expand(arg.coeff(momentum)/sympy.I)
               for momentum in momenta]
        # We do not allow sympy symbols in the hopping, should
        # be numerical values only.
        if any([isinstance(item, sympy.symbol.Symbol)
                        for ele in hop for item in ele.atoms()
                        if isinstance(ele, sympy.Expr)]):
            raise ValueError(
                "Real space part of the hopping must be numbers, not symbols."
            )
        # If the exponential contains something extra other than the
        # hopping part, we append it to the coefficient.
        spatial_arg = sympy.I*sum([ele*momentum for ele, momentum in zip(momenta, hop)])
        diff = sympy.nsimplify(sympy.expand(arg - spatial_arg))
        coeff = sympy.simplify(coeff * e**diff)
        hop = np.array(hop).astype(float)
    # Make sure there is no momentum dependence in the coefficient.
    if any([momentum in coeff.atoms() for momentum in momenta]):
        raise ValueError(
            "All momentum dependence should be confined to hopping exponentials."
        )
    return BlochCoeff(hop, coeff)


@lru_cache()
def _find_momenta(momenta):
    if any(isinstance(i, int) for i in momenta):
        raise TypeError('Momenta should be strings or sympy symbols.')
    elif all(m in _commutative_momenta for m in momenta):
        return tuple(momenta)
    else:
        _momenta = [kwant_continuum.sympify(k) for k in momenta]
        return tuple(kwant_continuum.make_commutative(k, k)
                        for k in _momenta)


@lru_cache(maxsize=1000)
def _symbol_normalizer(key):
    return sympy.expand_power_exp(sympy.sympify(key))


@lru_cache(maxsize=1000)
def _bloch_normalizer(key, momenta):
    if isinstance(key, BlochCoeff):
        return key
    else:
        return _to_bloch_coeff(key, momenta)


def _shape_and_format(val):
    # Find shape and type of val
    format = type(val)
    try:
        shape = val.shape
    except AttributeError:
        # Treat it as a scalar
        shape = ()
    if issubclass(format, Number):
        # Cast all numbers to np.complex128
        format = np.complex128
    elif issubclass(format, np.ndarray):
        format = np.ndarray
    elif issubclass(format, scipy.sparse.linalg.LinearOperator):
        # Make all subclasses of LinearOperator work
        format = scipy.sparse.linalg.LinearOperator
    elif not issubclass(format, scipy.sparse.spmatrix):
        raise ValueError('Only `formats` which are subclasses of `np.ndarray`, `scipy.sparse.spmatrix` '
                         '`scipy.sparse.linalg.LinearOperator` or `Number` are supported.')
    return shape, format
