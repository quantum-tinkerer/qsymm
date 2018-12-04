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
        symbolic substitutions, differentiation, etc."""
        if self == {}:
            result = self.zeros_like()
        else:
            # Add possible duplicate keys that only differ in constant factors
            result = sum(type(self)({func(key): val}, momenta=self.momenta)
                         for key, val in self.items())
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

        momenta = self.momenta
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
        result = type(substituted)({}, momenta=momenta)
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
                result += type(substituted)({rest * np.prod(expos): value}, momenta=momenta)
            else:
                result += type(substituted)({key: value}, momenta=momenta)
        return result

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
            result = sympy.sympify(sum(key * val for key, val in self.data.items()))
        else:
            # Vectorize nsimplify
            vnsimplify = np.vectorize(sympy.nsimplify, otypes=[object])
            result = sympy.MatAdd(*[key * sympy.Matrix(vnsimplify(val))
                                    for key, val in self.data.items()]).doit()
        if any([isinstance(result, matrix_type) for matrix_type in (sympy.MatrixBase,
                                                                    sympy.ImmutableDenseMatrix,
                                                                    sympy.ImmutableDenseNDimArray)]):
            result = sympy.Matrix(result).reshape(*result.shape)
        return result

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
            # This works if some keys are different but close, such that BlochCoeff
            # is the same
            self.__init__({}, momenta=hamiltonian.momenta)
            data = defaultdict(lambda: np.zeros(hamiltonian.shape, dtype=complex))
            for key, val in hamiltonian.items():
                data[_to_bloch_coeff(key, hamiltonian.momenta)] += val
            self.data = data
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
            elif symbolic:
                self.__init__(Model(hamiltonian, locals, momenta))
        else:
            # Use Model to parse input
            self.__init__(Model(hamiltonian, locals, momenta))
        self.shape = _find_shape(self.data)

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
        result = BlochModel({BlochCoeff(-hop, coeff.subs(sympy.I, -sympy.I)): val.conj()
                            for (hop, coeff), val in self.items()})
        result.momenta = self.momenta
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
        shape = next(iter(data.values())).shape
        if not all([v.shape == shape for v in data.values()]):
            raise ValueError('All terms must have the same shape')
        return shape
