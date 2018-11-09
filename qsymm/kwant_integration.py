import numpy as np
import scipy.linalg as la
import itertools as it
import kwant
from kwant._common import get_parameters
# XXX import voronoi when new version gets merged into stable
from kwant.linalg.lll import lll #, voronoi

import qsymm
from qsymm.model import HoppingCoeff
from qsymm.groups import generate_group, PointGroupElement, L_matrices, spin_rotation
from qsymm.linalg import allclose, prop_to_id
from qsymm.hamiltonian_generator import hamiltonian_from_family


def builder_to_model(syst, momenta=None):
    """Convert a kwant.Builder to a qsymm.Model

    Parameters
    ----------

    syst: kwant.Builder
        Kwant system to be turned into Model. Has to be an unfinalized
        Builder. Can have translation in any dimension.
    momenta: list of strings or None
        Names of momentum variables, if None 'k_x', 'k_y', ... is used.

    Returns:
    --------

    qsymm.Model
        Model representing the tight-binding Hamiltonian.
    """
    def term_to_model(d, par, matrix):
        if np.allclose(matrix, 0):
            result = qsymm.Model({})
        else:
            result = qsymm.Model({HoppingCoeff(d, qsymm.sympify(par)): matrix}, momenta=momenta)
        return result

    def hopping_to_term(hop, value):
        site1, site2 = hop
        d = proj @ np.array(site2.pos - site1.pos)
        slice1, slice2 = slices[to_fd(site1)], slices[to_fd(site2)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice2, val))
                       for par, val in function_to_terms(hop, value))
        else:
            matrix = set_block(slice1, slice2, value)
            return term_to_model(d, '1', matrix)

    def onsite_to_term(site, value):
        d = np.zeros((dim, ))
        slice1 = slices[to_fd(site)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice1, val))
                       for par, val in function_to_terms(site, value))
        else:
            return term_to_model(d, '1', set_block(slice1, slice1, value))

    def function_to_terms(site_or_hop, value):
        assert callable(value)
        parameters = get_parameters(value)
        if isinstance(site_or_hop, kwant.builder.Site):
            parameters = parameters[1:]
            site_or_hop = (site_or_hop,)
        else:
            parameters = parameters[2:]
        h_0 = value(*site_or_hop, *((0,) * len(parameters)))
        all_args = np.eye(len(parameters))
        
        terms = []
        for p, args in zip(parameters, all_args):
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
    if len(periods) == 0:
        proj = np.empty((0, len(list(syst.sites())[0].pos)))
    else:
        proj, r = la.qr(np.array(periods).T, mode='economic')
        sign = np.diag(np.diag(np.sign(r)))
        proj = sign @ proj.T

    slices, N = orbital_slices(syst)
    
    one_way_hoppings = [hopping_to_term(hop, value) for hop, value in syst.hopping_value_pairs()]
    hoppings = one_way_hoppings + [term.T().conj() for term in one_way_hoppings]
    
    onsites = [onsite_to_term(site, value) for site, value in syst.site_value_pairs()]
    
    result = sum(onsites) + sum(hoppings)
    return result

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
        key = HoppingCoeff(signature, one)
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
    neighbors = {HoppingCoeff(vec, one) for vec in neighbors}
    for g in group:
        r_neighbors = {HoppingCoeff(g.R @ hop, coeff) for (hop, coeff) in neighbors}
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