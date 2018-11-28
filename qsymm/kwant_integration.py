import numpy as np
import tinyarray as ta
import scipy.linalg as la
import itertools as it
import kwant
import sympy
from collections import OrderedDict, defaultdict
from kwant._common import get_parameters
# XXX import voronoi when new version gets merged into stable
from kwant.linalg.lll import lll #, voronoi

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


def bloch_model_to_builder(model, norbs, lat_vecs, atom_coords):
    """Make a kwant builder out of a Model object"""
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


def bloch_family_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None):
    """Make a kwant builder from a family of Bloch Hamiltonians."""
    ham = hamiltonian_from_family(family, coeffs=coeffs, nsimplify=False, tosympy=False)
    return bloch_model_to_builder(ham, norbs, lat_vecs, atom_coords)


def kp_to_builder(family, coeffs=None, nsimplify=True, coords=None, *, grid=None, locals=None):
    """Make a discretized Kwant builder from a Model representing a continuum k.p
    Hamiltonian. """
    ham = hamiltonian_from_family(family, coeffs=coeffs, nsimplify=True)
    builder = kwant.continuum.discretize(ham, coords=coords, grid=grid, locals=locals)
    return builder


def bravais_point_group(periods, tr=True, ph=True, generators=False, verbose=False):
    """Find the  point group of the Bravais-lattice defined by periods.

    Parameters:
    -----------
    periods: array
        Translation vectors as row vectors, arranged into a 2D array.
    tr, ph: bool (default True)
        Whether to return time reversal and particle-hole operators.
        If false, only pure point-group operators are returned.
    generators: bool (default False)
        If True only a (not necessarily minimal) generator set of the
        symmetry group is returned.
    verbose: bool (default False)
        If True, the name of the Bravais lattice is printed.

    Returns:
    --------
    set of PointGroupElements
    """
    periods = np.asarray(periods)
    dim = periods.shape[0]
    # Project onto the subspace spanned by the translations
    if periods.shape[1] > periods.shape[0]:
        proj, r = la.qr(periods.T, mode='economic')
        sign = np.diag(np.diag(np.sign(r)))
        proj = sign @ proj.T
        periods = periods @ proj.T

    # get lll reduced basis
    periods, _ = lll(periods)
    # get nearest neighbors
    neighbors = voronoi(periods, reduced=True, rtol=1e-5) @ periods
    neighbors = neighbors[:len(neighbors)//2]
    num_eq, sets_eq = equals(neighbors)

    # Everey Bravais-lattice group contains inversion
    gens = {PointGroupElement(-np.eye(dim))}

    if dim==1:
        # The only bravais lattice group in 1D only contains inversion
        pass
    elif dim==2:
        gens |= bravais_group_2d(neighbors, num_eq, sets_eq, verbose)
    elif dim==3:
        gens |= bravais_group_3d(neighbors, num_eq, sets_eq, verbose)
    else:
        raise NotImplementedError('Only 1, 2, and 3 dimensional translation symmetry supported.')

    if tr:
        TR = PointGroupElement(np.eye(dim), True, False, None)
        gens.add(TR)
    if ph:
        PH = PointGroupElement(np.eye(dim), True, True, None)
        gens.add(PH)
    assert check_bravais_symmetry(neighbors, gens)
    if generators:
        return gens
    else:
        return generate_group(gens)


def bravais_group_2d(neighbors, num_eq, sets_eq, verbose=False):
    s3 = np.sqrt(3)
    gens = set()

    assert len(neighbors) <= 3
    if num_eq == [1, 1] or num_eq == [2]:
        # Sqare or simple rectangular lattice
        name = 'simple rectangular'
        # Mirrors
        Mx = PointGroupElement(mirror(neighbors[0]))
        My = PointGroupElement(mirror(neighbors[1]))
        gens |= {Mx, My}
        if num_eq == [2]:
            # Square lattice, 4-fold rotation
            name = 'square'
            C4 = PointGroupElement(np.array([[0, 1], [-1, 0]]))
            gens.add(C4)
    elif num_eq == [1, 2] or num_eq == [3]:
        # Centered rectangular, mirror symmetry
        name = 'centered rectangular'
        vecs = sets_eq[-1][:2]
        Mx = PointGroupElement(mirror(vecs[0] + vecs[1]))
        My = PointGroupElement(mirror(vecs[0] - vecs[1]))
        gens.add(Mx)
        gens.add(My)
        if num_eq == [3]:
            name = 'hexagonal'
            # Hexagonal lattice, 6-fold rotation
            C6 = PointGroupElement(np.array([[1/2, s3/2], [-s3/2, 1/2]]))
            gens.add(C6)
    else:
        name = 'oblique'

    if verbose:
        print(name)
    return gens


def bravais_group_3d(neighbors, num_eq, sets_eq, verbose=False):
    assert len(neighbors) <= 7
    gens = set()

    if num_eq == [3]:
        # Primitive cubic, 3 4-fold axes
        name = 'primitive cubic'
        C4s = {rotation(n, 4) for n in neighbors}
        gens |= C4s
    elif num_eq == [1, 2]:
        # Primitive tetragonal, find 4-fold axis
        name = 'primitive tetragonal'
        C4 = rotation(np.cross(*sets_eq[1]), 4)
        gens.add(C4)
        C2s = {rotation(axis, 2) for axis in sets_eq[1]}
        gens |= C2s
    elif num_eq == [1, 1, 1]:
        # Primitive orthorhombic
        name = 'primitive orthorhombic'
        C2s = {rotation(n, 2) for n in neighbors}
        gens |= C2s
    elif num_eq == [1, 3]:
        # Hexagonal
        name = 'hexagonal'
        # lone one for C6 axis
        C6 = rotation(sets_eq[0][0], 6)
        # one of the triple for C2 axis
        C2 = rotation(sets_eq[1][0], 2)
        gens |= {C6, C2}
    elif num_eq == [1, 1, 2]:
        # Base centered orthorhombic
        name = 'base centered orthorhombic'
        vec011, vec01m1 = sets_eq[-1]
        C2x = rotation(vec011 + vec01m1, 2)
        C2y = rotation(vec011 - vec01m1, 2)
        C2z = rotation(np.cross(vec011, vec01m1), 2)
        gens |= {C2x, C2y, C2z}
    elif num_eq == [1, 1, 1, 1]:
        # Primitive Monoclinic
        name = 'primitive monoclinic'
        axis, = pick_perp(neighbors, 3)
        C2 = rotation(axis, 2)
        gens.add(C2)
    elif num_eq == [3, 4]:
        # Body centered cubic
        name = 'body centered cubic'
        C4s = {rotation(n, 4) for n in sets_eq[0]}
        gens |= C4s
    elif num_eq == [1, 2, 4] or num_eq == [2, 4]:
        # Body centered tetragonal
        name = 'body centered tetragonal'
        axes = (sets_eq[0] if len(num_eq) == 2 else sets_eq[1])
        C2s = {rotation(n, 2) for n in axes}
        C4 = rotation(np.cross(*axes), 4)
        gens |= C2s
        gens.add(C4)
    elif num_eq == [1, 1, 1, 4] or num_eq == [1, 1, 4]:
        # Body centered orthorhombic
        name = 'body centered orthorhombic'
        axes = [vec[0] for num, vec in zip(num_eq, sets_eq) if num == 1]
        C2s = {rotation(n, 2) for n in axes}
        C2s.add(rotation(np.cross(axes[0], axes[1]), 2))
        gens |= C2s
    elif num_eq == [6]:
        # FCC
        name = 'face centered cubic'
        # pick an orthogonal pair to define C4 axes
        vec110 = neighbors[0]
        vec1m10, = pick_perp(neighbors, 1, [vec110])
        C4x = rotation(vec110 + vec1m10, 4)
        C4y = rotation(vec110 - vec1m10, 4)
        C4z = rotation(np.cross(vec110, vec1m10), 4)
        gens |= {C4x, C4y, C4z}
    elif num_eq == [1, 3, 3] or num_eq == [3, 3]:
        # Rhombohedral with or without face toward the 3-fold axis
        name = 'rhombohedral'
        C3 = threefold_axis(sets_eq[-1], neighbors)
        assert C3 is not None, sets_eq
        C2 = twofold_axis(sets_eq[-1], neighbors)
        assert C2 is not None, sets_eq
        gens |= {C3, C2}
    elif num_eq == [1, 2, 2, 2]:
        # Face centered orthorhombic
        name = 'face centered orthorhombic'
        # One cubic vector has to be there
        vec100 = sets_eq[0][0]
        C2x = rotation(vec100, 2)
        # Pick the pair orthogonal to it
        vec011, vec01m1 = pick_perp(neighbors, 1, [vec100])
        C2y = rotation(vec011 + vec01m1, 2)
        C2z = rotation(vec011 - vec01m1, 2)
        gens |= {C2x, C2y, C2z}
    elif num_eq[-1] == num_eq[-2] == 2:
        # Base centered monoclinic
        name = 'base centered monoclinic'
        # some combination of the equal pairs has to be 2-fold axis
        for vecs in sets_eq[-2:]:
            C2 = twofold_axis(vecs, neighbors)
            assert C2 is not None
            gens.add(C2)
    else:
        assert max(num_eq) == 1
        name = 'triclinic'

    if verbose:
        print(name)
    return gens


def mirror(n):
    n = np.array(n)
    return np.eye(len(n)) - 2 * np.outer(n, n) / (n @ n)


def rotation(normal, fold):
    n = 2 * np.pi / fold * np.array(normal) / la.norm(normal)
    Ls = L_matrices()
    return PointGroupElement(np.real(spin_rotation(n, Ls)))


def equals(vectors):
    # group equivalent vectors based on length and angles
    one = qsymm.sympify('1')
    sets = dict()
    # Take abs because every vector has opposite pair
    overlaps = np.abs(vectors @ vectors.T)
    angles = np.outer(np.diag(overlaps), np.diag(overlaps))**(-1/2) * overlaps
    for i, vector in enumerate(vectors):
        length = np.array([overlaps[i, i]])
        # Symmetry equivalent vectors must have the same signature
        signature = np.concatenate([length, sorted(overlaps[i]), sorted(angles[i])])
        key = BlochCoeff(signature, one)
        if key in sets:
            sets[key].append(vector)
        else:
            sets[key] = [vector]
    numbers, sets = zip(*((len(set), np.array(set)) for set in sets.values()))
    order = np.argsort(numbers)
    numbers = np.array(numbers)
    sets = np.array(sets)
    return list(numbers[order]), list(sets[order])


def pick_perp(vectors, n, other_vectors=None):
    # Pick vectors that are orthogonal to at least n other vectors
    other_vectors = np.array(vectors if other_vectors is None else other_vectors)
    perp = []
    non_perp = []
    for v in vectors:
        if np.sum(np.isclose(v @ other_vectors.T, 0)) >= n:
            perp.append(v)
    return perp


def threefold_axis(vectors, neighbors):
    # Find threefold axis that leaves three vectors invariant
    assert len(vectors) == 3
    overlaps = vectors @ vectors.T
    prop, norm = prop_to_id(np.diag(np.diag(overlaps)))
    if not prop:
        return None
    overlaps = np.abs(overlaps) / norm
    if np.allclose([overlaps[0, 1], overlaps[0, 2], overlaps[1, 2]], 1/2):
        # coplanar vectors, may be 6-fold
        axis = np.cross(vectors[0], vectors[1])
        C3 = rotation(axis, 3)
        C6 = rotation(axis, 6)
        if check_bravais_symmetry(neighbors, {C6}):
            return C6
        elif check_bravais_symmetry(neighbors, {C3}):
            return C3
        else:
            return None
    for signs in it.product([1, -1], repeat=3):
        axis = signs @ vectors
        overlaps = axis @ vectors.T
        C3 = rotation(axis, 3)
        prop, _ = prop_to_id(np.diag(np.abs(overlaps)))
        if prop and check_bravais_symmetry(neighbors, {C3}):
            return C3
    else:
        return None


def twofold_axis(vectors, neighbors):
    # Find twofold axis from vectors that leaves neighbors invariant
    for sign, (vec1, vec2) in it.product([1, -1], it.combinations(vectors, 2)):
        axis = vec1 + sign * vec2
        C2 = rotation(axis, 2)
        if check_bravais_symmetry(neighbors, {C2}):
            return C2
    else:
        return None


def check_bravais_symmetry(neighbors, group):
    one = qsymm.sympify('1')
    neighbors = np.vstack([neighbors, -neighbors])
    neighbors = {BlochCoeff(vec, one) for vec in neighbors}
    for g in group:
        r_neighbors = {BlochCoeff(g.R @ hop, coeff) for (hop, coeff) in neighbors}
        if not neighbors == r_neighbors:
            return False
    else:
        return True


# XXX This version of voronoi is in the latest build of kwant, remove when it is merged into stable
from kwant.linalg.lll import cvp

def voronoi(basis, reduced=False, rtol=1e-09):
    """
    Return an array of lattice vectors forming its voronoi cell.

    Parameters
    ----------
    basis : 2d array-like of floats
        Basis vectors for which the Voronoi neighbors have to be found.
    reduced : bool
        If False, exactly `2 (2^D - 1)` vectors are returned (where `D`
        is the number of lattice vectors), these are not always the minimal
        set of Voronoi vectors.
        If True, only the minimal set of Voronoi vectors are returned.
    rtol : float
        Tolerance when deciding whether a vector is in the minimal set,
        vectors associated with small faces compared to the size of the
        unit cell are discarded.

    Returns
    -------
    voronoi_neighbors : numpy array of ints
        All the lattice vectors that (potentially) neighbor the origin.
        List of length `2n` where the second half of the list contains
        is `-1` times the vectors in the first half.
    """
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError('`basis` must be a 2d array-like object.')
    # Find halved lattice points, every face of the VC contains half
    # of the lattice vector which is the normal of the face.
    # These points are all potentially on a face a VC,
    # but not necessarily the VC centered at the origin.
    displacements = list(it.product(*(len(basis) * [[0, .5]])))[1:]
    # Find the nearest lattice point, this is the lattice point whose
    # VC this face belongs to.
    vertices = np.array([cvp(vec @ basis, basis)[0] for vec in
                         displacements])
    # The lattice vector for a face is exactly twice the vector to the
    # closest lattice point from the halved lattice point on the face.
    vertices = np.array(np.round((vertices - displacements) * 2), int)
    if reduced:
        # Discard vertices that are not associated with a face of the VC.
        # This happens if half of the lattice vector is outside the convex
        # polytope defined by the rest of the vertices in `keep`.
        bbt = basis @ basis.T
        products = vertices @ bbt @ vertices.T
        keep = np.array([True] * len(vertices))
        for i in range(len(vertices)):
            # Relevant if the projection of half of the lattice vector `vertices[i]`
            # onto every other lattice vector in `veritces[keep]` is less than `0.5`.
            mask = np.array([False if k == i else b for k, b in enumerate(keep)])
            projections = 0.5 * products[i, mask] / np.diag(products)[mask]
            if not np.all(np.abs(projections) < 0.5 - rtol):
                keep[i] = False
        vertices = vertices[keep]

    vertices = np.concatenate([vertices, -vertices])
    return vertices