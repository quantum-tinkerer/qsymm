import numpy as np
import tinyarray as ta
import scipy.linalg as la
import itertools as it
import kwant
import sympy
from collections import OrderedDict, defaultdict
from kwant._common import get_parameters

import qsymm
from qsymm.model import Model, BlochModel, BlochCoeff, _commutative_momenta
from qsymm.groups import generate_group, PointGroupElement, L_matrices, \
                         spin_rotation, ContinuousGroupGenerator
from qsymm.linalg import allclose, prop_to_id
from qsymm.hamiltonian_generator import hamiltonian_from_family


def builder_to_model(syst, momenta=None, unit_cell_convention=False, params=dict()):
    """Convert a kwant.Builder to a qsymm.BlochModel

    Parameters
    ----------

    syst: kwant.Builder
        Kwant system to be turned into BlochModel. Has to be an unfinalized
        Builder. Can have translation in any dimension.
    momenta: list of strings or None
        Names of momentum variables, if None 'k_x', 'k_y', ... is used.
    unit_cell_convention: bool (default False)
        If True, use the unit cell convention for Bloch basis, the
        exponential has the difference in the unit cell coordinates and
        k is expressed in the reciprocal lattice basis. This is consistent
        with kwant.wraparound.
        If False, the difference in the real space coordinates is used
        and k is given in an absolute basis.
        Only the default choice guarantees that qsymm is able to find
        nonsymmorphic symmetries.
    params: dict (default empty)
        Dictionary of parameter: value to substitute in the builder.

    Returns:
    --------

    qsymm.BlochModel
        Model representing the tight-binding Hamiltonian.
    """
    def term_to_model(d, par, matrix):
        if np.allclose(matrix, 0):
            result = BlochModel({})
        else:
            result = BlochModel({BlochCoeff(d, qsymm.sympify(par)): matrix}, momenta=momenta)
        return result

    def hopping_to_model(hop, value, proj, params):
        site1, site2 = hop
        if unit_cell_convention:
            # same as site2.tag - site1.tag if there is only one lattice site in the FD
            d = np.array(syst.symmetry.which(site2))
        else:
            d = proj @ np.array(site2.pos - site1.pos)
        slice1, slice2 = slices[to_fd(site1)], slices[to_fd(site2)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice2, val))
                       for par, val in function_to_terms(hop, value, params))
        else:
            matrix = set_block(slice1, slice2, value)
            return term_to_model(d, '1', matrix)

    def onsite_to_model(site, value, params):
        d = np.zeros((dim, ))
        slice1 = slices[to_fd(site)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice1, val))
                       for par, val in function_to_terms(site, value, params))
        else:
            return term_to_model(d, '1', set_block(slice1, slice1, value))

    def function_to_terms(site_or_hop, value, fixed_params):
        assert callable(value)
        parameters = get_parameters(value)
        # remove site or site1, site2 parameters
        if isinstance(site_or_hop, kwant.builder.Site):
            parameters = parameters[1:]
            site_or_hop = (site_or_hop,)
        else:
            parameters = parameters[2:]
        free_parameters = (par for par in parameters if par not in fixed_params.keys())
        # first set all free parameters to 0
        args = ((fixed_params[par] if par in fixed_params.keys() else 0) for par in parameters)
        h_0 = value(*site_or_hop, *args)
        # set one of the free parameters to 1 at a time, the rest 0
        terms = []
        for p in free_parameters:
            args = ((fixed_params[par] if par in fixed_params.keys() else
                     (1 if par == p else 0)) for par in parameters)
            terms.append((p, value(*site_or_hop, *args) - h_0))
        return terms + [('1', h_0)]

    def orbital_slices(syst):
        orbital_slices = {}
        start_orb = 0

        for site in syst.sites():
            n = site.family.norbs
            if n is None:
                raise ValueError('norbs must be provided for every lattice.')
            orbital_slices[site] = slice(start_orb, start_orb + n)
            start_orb += n
        return orbital_slices, start_orb
    
    def set_block(slice1, slice2, val):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[slice1, slice2] = val
        return matrix
    
    periods = np.array(syst.symmetry.periods)
    dim = len(periods)
    to_fd = syst.symmetry.to_fd
    if momenta is None:
        momenta = ['k_x', 'k_y', 'k_z'][:dim]
    # If the system is higher dimensional than the numder of translation vectors, we need to
    # project onto the subspace spanned by the translation vectors.
    if dim == 0:
        proj = np.empty((0, len(list(syst.sites())[0].pos)))
    elif dim < len(list(syst.sites())[0].pos):
        proj, r = la.qr(np.array(periods).T, mode='economic')
        sign = np.diag(np.diag(np.sign(r)))
        proj = sign @ proj.T
    else:
        proj = np.eye(dim)

    slices, N = orbital_slices(syst)

    one_way_hoppings = [hopping_to_model(hop, value, proj, params) for hop, value in syst.hopping_value_pairs()]
    hoppings = one_way_hoppings + [term.T().conj() for term in one_way_hoppings]

    onsites = [onsite_to_model(site, value, params) for site, value in syst.site_value_pairs()]

    result = sum(onsites) + sum(hoppings)
    
    return result


def builder_discrete_symmetries(builder, spatial_dimensions=3):
    """Extract the discrete symmetries of a Kwant builder.
    
    Parameters
    ----------
    builder: kwant.Builder
        An unfinalized Kwant system.
    spatial_dimensions: integer, default 3
        Number of spatial dimensions in the system.
    
    Returns
    -------
    builder_symmetries: dict
        Dictionary of the discrete symmetries that the builder has.
        The symmetries can be particle-hole, time-reversal or chiral,
        which are returned as PointGroupElement objects, or
        a conservation law, which is returned as a ContinuousGroupGenerator
        object.
    """
    
    symmetry_names = ['time_reversal', 'particle_hole', 'chiral', 'conservation_law']
    builder_symmetries = {name: getattr(builder, name) for name in symmetry_names
                         if getattr(builder, name) is not None}
    for name, symmetry in builder_symmetries.items():
        if name is 'time_reversal':
            builder_symmetries[name] = PointGroupElement(np.eye(spatial_dimensions),
                                                         True, False, symmetry)
        elif name is 'particle_hole':
            builder_symmetries[name] = PointGroupElement(np.eye(spatial_dimensions),
                                                         True, True, symmetry)
        elif name is 'chiral':
            builder_symmetries[name] = PointGroupElement(np.eye(spatial_dimensions),
                                                         False, True, symmetry)
        elif name is 'conservation_law':
            builder_symmetries[name] = ContinuousGroupGenerator(R=None, U=symmetry)
        else:
            raise ValueError("Invalid symmetry name.")
    return builder_symmetries


def bloch_to_builder(model, norbs, lat_vecs, atom_coords, *, coeffs=None):
    """Make a Kwant builder out of a Model or BlochModel object representing
    a tight binding Hamiltonian in Bloch form, or from a list of such 
    Model or BlochModel objects.
    
    Parameters
    ----------
    model: qsymm.model.Model, qsymm.model.BlochModel, or list thereof.
        Object or objects representing the Hamiltonian to convert
        into a Kwant builder.
    norbs: OrderedDict : {site : norbs_site} or tuple of tuples ((site, norbs_site), )
        Sites and number of orbitals per site in a unit cell. 
    lat_vecs: list of arrays.
        Lattice vectors of the underlying tight binding lattice.
    atom_coords: list of arrays.
        Positions of the sites (or atoms) within a unit cell.
        The ordering of the atoms is the same as in norbs.
    coeffs: list of sympy.Symbol, default None.
        Constant prefactors for the individual terms in model, if model
        is a list of multiple objects. If model is a single Model or BlochModel
        object, this argument is ignored. By default assigns the coefficient
        c_n to element model[n].
    
    Returns
    -----------
    syst: kwant.Builder
        The unfinalized Kwant system representing the qsymm Model(s).
    
    Notes
    -----
    Onsite terms that are not provided in the input model are set
    to zero by default.
    
    """
    
    # If input is a list of Model objects, combine into a single object.
    if isinstance(model, list):
        model = hamiltonian_from_family(model, coeffs=coeffs,
                                        nsimplify=False, tosympy=False)
    # Convert to BlochModel
    if not isinstance(model, BlochModel):
        model = BlochModel(model)
    
    momenta = model.momenta
    assert len(momenta) == len(lat_vecs), "dimension of the lattice and number of momenta do not match"
    
    # Subblocks of the Hamiltonian for different atoms.
    N = 0
    norbs = OrderedDict(norbs)
    ranges = dict()
    for a, n in norbs.items():
        ranges[a] = slice(N, N + n)
        N += n
        
    # Extract atoms and number of orbitals per atom,
    # store the position of each atom
    atoms, orbs = zip(*[(atom, norb) for atom, norb in
                        norbs.items()])
    coords_dict = {atom: coord for atom, coord in
                   zip(atoms, atom_coords)}
    
    # Make the kwant lattice
    lat = kwant.lattice.general(lat_vecs,
                                atom_coords,
                                norbs=orbs)
    # Store sublattices by name
    sublattices = {atom: sublat for atom, sublat in
                   zip(atoms, lat.sublattices)}
    sym = kwant.TranslationalSymmetry(*lat_vecs)
    syst = kwant.Builder(sym)
    
    def make_int(R):
        # If close to an integer array convert to integer tinyarray, else
        # return None
        R_int = ta.array(np.round(R), int)
        if qsymm.linalg.allclose(R, R_int):
            return R_int
        else:
            return None
        
    # Keep track of the hoppings and onsites by storing those
    # which have already been set.
    hopping_dict = defaultdict(lambda: {})
    onsites_dict = defaultdict(lambda: {})
    
    zer = [0]*len(momenta)
    
    # Iterate over all items in the model.
    for key, hop_mat in model.items():
        # Determine whether this is an onsite or a hopping, extract
        # overall symbolic coefficient if any, extract the exponential
        # part describing the hopping if present.
        r_vec, coeff = key
        # Onsite term
        if allclose(r_vec, 0):
            for atom1, atom2 in it.product(atoms, atoms):
                # Subblock within the same sublattice is onsite
                hop = hop_mat[ranges[atom1], ranges[atom2]]
                if sublattices[atom1] == sublattices[atom2]:
                    onsite = Model({coeff: hop}, momenta=momenta)
                    onsites_dict[atom1] += onsite
                # Blocks between sublattices are hoppings between sublattices
                # at the same position.
                # Only include nonzero hoppings
                elif not allclose(hop, 0):
                    assert allclose(np.array(coords_dict[atom1]), np.array(coords_dict[atom2]))
                    lat_basis = np.array(zer)
                    hop = Model({coeff: hop}, momenta=momenta)
                    hop_dir = kwant.builder.HoppingKind(-lat_basis, sublattices[atom1], sublattices[atom2])
                    hopping_dict[hop_dir] += hop

        else:
            # Iterate over combinations of atoms, set hoppings between each
            for atom1, atom2 in it.product(atoms, atoms):
                # Take the block from atom1 to atom2
                hop = hop_mat[ranges[atom1], ranges[atom2]]
                # Only include nonzero hoppings
                if not allclose(hop, 0):
                    # Adjust hopping vector to Bloch form basis
                    r_lattice = r_vec + np.array(coords_dict[atom1]) - np.array(coords_dict[atom2])
                    # Bring vector to basis of lattice vectors
                    lat_basis = np.linalg.solve(np.vstack(lat_vecs).T, r_lattice)
                    lat_basis = make_int(lat_basis)
                    # Should only have hoppings that are integer multiples of lattice vectors
                    if lat_basis is not None:
                        hop_dir = kwant.builder.HoppingKind(-lat_basis, sublattices[atom1], sublattices[atom2])
                        hop = Model({coeff: hop}, momenta=momenta)
                        # Set the hopping as the matrix times the hopping amplitude
                        hopping_dict[hop_dir] += hop
                    else:
                        raise RunTimeError('A nonzero hopping not matching a lattice vector was found.')
            
    # If some onsite terms are not set, we set them to zero.
    for atom in atoms:
        if atom not in onsites_dict:
            onsites_dict[atom] = Model({sympy.numbers.One(): np.zeros((norbs[atom], norbs[atom]))},
                                             momenta=momenta)
            
    # Iterate over all onsites and set them
    for atom, onsite in onsites_dict.items():
        # works, but surely there is a better way
        syst[sublattices[atom](*zer)] = onsite.lambdify(onsite=True)
        
    # Finally, iterate over all the hoppings and set them
    for direction, hopping in hopping_dict.items():
        syst[direction] = hopping.lambdify(hopping=True)

    return syst


def kp_to_builder(model, coords=None, *, grid=None, locals=None, coeffs=None):
    """Make a Kwant builder out of a Model object representing a continuum
    k.p Hamiltonian, or a list of such Model objects, by discretizing
    the continuum Hamiltonian.
    
    Parameters
    ----------
    model: qsymm.model.Model or list of qsymm.model.Model objects.
        Object or objects representing the Hamiltonian to convert
        into a Kwant builder.
    coords : sequence of strings, optional
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.  If not provided, `coords` will be obtained from the input
        Hamiltonian by reading the present coordinates and momentum operators.
    grid : scalar or kwant.lattice.Monatomic instance, optional
        Lattice that will be used as a discretization grid. It must have
        orthogonal primitive vectors. If a scalar value is given, a lattice
        with the appropriate grid spacing will be generated. If not provided,
        a lattice with grid spacing 1 in all directions will be generated.
    locals : dict, optional
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
    grid_spacing : int or float, optional
        (deprecated) Spacing of the discretization grid. If unset the default
        value will be 1. Cannot be used together with ``grid``.
    coeffs: list of sympy.Symbol, default None.
        Constant prefactors for the individual terms in model, if model
        is a list of multiple objects. If model is a single Model or BlochModel
        object, this argument is ignored. By default assigns the coefficient
        c_n to element model[n].
    
    Returns
    -----------
    syst: kwant.Builder
        The unfinalized Kwant system representing the qsymm Model(s).
    
    Notes
    -----
    This function discretizes the continuum Hamiltonian to form a tight binding
    model, by calling kwant.continuum.discretize.
    
    """
    
    # If input is a list of Model objects, combine them into one.
    if isinstance(model, list):
        model = hamiltonian_from_family(model, coeffs=coeffs)
    
    ham = model.tosympy(nsimplify=True)
    if any([isinstance(ham, matrix_type) for matrix_type in
            (sympy.MatrixBase, sympy.ImmutableDenseMatrix,
             sympy.ImmutableDenseNDimArray)]):
        ham = sympy.Matrix(ham)
    return kwant.continuum.discretize(kwant.continuum.sympify(ham),
                                      coords=coords, grid=grid, locals=locals)
