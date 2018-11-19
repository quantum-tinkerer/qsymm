import pytest
import numpy as np
import sympy
import kwant
import warnings
from collections import OrderedDict

from ..symmetry_finder import symmetries
from ..hamiltonian_generator import bloch_family, hamiltonian_from_family
from ..groups import hexagonal, PointGroupElement
from ..model import Model
from ..kwant_integration import builder_to_model, bravais_point_group, \
                                bloch_model_to_builder, bloch_family_to_builder


def _version_higher(v='1.4.0'):
    from kwant import __version__ as n
    v = tuple(int(char) for char in  v[:5].split('.'))
    n = tuple(int(char) for char in  n[:5].split('.'))
    if n >= v:
        return True
    return False


def test_honeycomb():
    try:
        assert _version_higher(v='1.4.0'), 'needs kwant >= 1.4.0'
    except AssertionError:
        warnings.warn('Running kwant version < 1.4.0, skipping honeycomb test'
                      'which requires kwant >= 1.4.0.')
        return True

    lat = kwant.lattice.honeycomb(norbs=1)

    # Test simple honeycomb model with constant terms
    # Add discrete symmetries to the kwant builder as well, to check that they are
    # returned as well.
    syst = kwant.Builder(symmetry=kwant.lattice.TranslationalSymmetry(*lat.prim_vecs),
                         particle_hole=np.eye(2),
                         conservation_law=2*np.eye(2))
    syst[lat.a(0, 0)] = 1
    syst[lat.b(0, 0)] = 1
    syst[lat.neighbors(1)] = -1

    H, builder_symmetries = builder_to_model(syst)
    assert len(builder_symmetries) == 2
    assert np.allclose(builder_symmetries['particle_hole'], np.eye(2))
    assert np.allclose(builder_symmetries['conservation_law'], 2*np.eye(2))
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 24
    assert len(cs) == 0

    # Test simple honeycomb model with value functions
    syst = kwant.Builder(symmetry=kwant.lattice.TranslationalSymmetry(*lat.prim_vecs))
    syst[lat.a(0, 0)] = lambda site, ma: ma
    syst[lat.b(0, 0)] = lambda site, mb: mb
    syst[lat.neighbors(1)] = lambda site1, site2, t: t

    H, builder_symmetries = builder_to_model(syst)
    assert len(builder_symmetries) == 0
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 12
    assert len(cs) == 0

def test_higher_dim():
    # Test 0D finite system
    lat = kwant.lattice.cubic(norbs=1)
    syst = kwant.Builder()
    syst[lat(0, 0, 0)] = 1
    syst[lat(1, 1, 0)] = 1
    syst[lat(0, 1, 1)] = 1
    syst[lat(1, 0, -1)] = 1
    syst[lat(0, 0, 0), lat(1, 1, 0)] = -1
    syst[lat(0, 0, 0), lat(0, 1, 1)] = -1
    syst[lat(0, 0, 0), lat(1, 0, -1)] = -1

    H, builder_symmetries = builder_to_model(syst)
    assert len(builder_symmetries) == 0
    sg, cs = symmetries(H, prettify=True)
    assert len(sg) == 2
    assert len(cs) == 5

    # Test triangular lattice system embedded in 3D
    sym = kwant.lattice.TranslationalSymmetry([1, 1, 0], [0, 1, 1])
    lat = kwant.lattice.cubic(norbs=1)
    syst = kwant.Builder(symmetry=sym)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 0), lat(1, 1, 0)] = -1
    syst[lat(0, 0, 0), lat(0, 1, 1)] = -1
    syst[lat(0, 0, 0), lat(1, 0, -1)] = -1

    H, builder_symmetries = builder_to_model(syst)
    assert len(builder_symmetries) == 0
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 24
    assert len(cs) == 0

def test_bravais_symmetry():
    # 2D
    # random rotation
    R = kwant.rmt.circular(2, sym='D', rng=1)
    lattices = [
                ([[np.sqrt(3)/2, 1/2], [0, 1]], 12, 'hexagonal'),
                ([[1, 0], [0, 1]], 8, 'square'),
                ([[-1/3, 1], [1/3, 1]], 4, 'centered orthorhombic')
                ]
    for periods, n, name in lattices:
        group = bravais_point_group(periods, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)
        group = bravais_point_group(periods @ R, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)

    # 3D
    # random rotation
    R = kwant.rmt.circular(3, sym='D', rng=1)
    lattices = [
                (np.eye(3), 48, 'primitive cubic'),
                ([[1, 0, 0], [0, 1, 0], [1/2, 1/2, 1/2]], 48, 'BCC'),
                ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], 48, 'FCC'),
                ([[1, 0, 0], [0, 1, 0], [0, 0, 3]], 16, 'primitive tetragonal'),
                ([[1, 0, 0], [0, 1, 0], [1/2, 1/2, 3]], 16, 'body centered tetragonal'),
                ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], 8, 'primitive orthorhombic'),
                ([[1, 0, 0], [0, 2, 0], [1/2, 1, 3]], 8, 'body centered orthorhombic'),
                ([[10, 3, 0], [10, 0, 4], [0, 3, 4]], 8, 'face centered orthorhombic'),
                ([[1, 3, 0], [1, 0, 4], [0, 3, 4]], 8, 'face centered orthorhombic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [0, 0, 4]], 8, 'base centered orthorhombic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [0, 0, np.sqrt(10)/3]], 8, 'base centered orthorhombic corner case'),
                ([[1, 0, 0], [0, 2, 0], [1/10, 0, 4]], 4, 'primitive monoclinic'),
                ([[1, 0, 0], [0, 1, 0], [0, 1/10, 2]], 4, 'primitive monoclinic'),
                ([[1, 1/3, 0], [1, -1/3, 0], [1/10, 0, 4]], 4, 'base centered monoclinic'),
                ([[3, 0, 1], [3/2, 1, 1/2], [0, 0, 4]], 4, 'base centered monoclinic'),
                ([[3, 0, 1], [3/2, 1/2, 1/2], [0, 0, 4]], 4, 'base centered monoclinic'),
                ([[1, 0, 1/10], [0, 2, 1/10], [0, 0, 3]], 2, 'triclinic'),
                ([[1, 0, 1/5], [-1/2, np.sqrt(3)/2, 1/5], [-1/2, -np.sqrt(3)/2, 1/5]], 12, 'rhombohedral'),
                ([[1, 0, 5], [-1/2, np.sqrt(3)/2, 5], [-1/2, -np.sqrt(3)/2, 5]], 12, 'rhombohedral'),
                ([[np.sqrt(3)/2, 1/2, 0], [0, 1, 0], [0, 0, 2]], 24, 'hexagonal'),
                ([[np.sqrt(3)/2, 1/2, 0], [0, 1, 0], [0, 0, 1]], 24, 'hexagonal equal length corner case'),
                ]
    for periods, n, name in lattices:
        group = bravais_point_group(periods, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)
        group = bravais_point_group(periods @ R, tr=False, ph=False)
        assert len(group) == n, (name, periods, group, n)

        
def test_graphene_to_kwant():
    
    norbs = OrderedDict({'A': 1, 'B': 1})  # A and B atom per unit cell, one orbital each
    hopping_vectors = [('A', 'B', [1, 0])] # Hopping between neighbouring A and B atoms
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0), (1, 0)]
    # We set the interatom distance to 1, so the lattice vectors have length sqrt(3)
    lat_vecs = [(3/2, np.sqrt(3)/2), (3/2, -np.sqrt(3)/2)]
    
    # Time reversal
    TR = PointGroupElement(sympy.eye(2), True, False, np.eye(2))
    # Chiral symmetry
    C = PointGroupElement(sympy.eye(2), False, True, np.array([[1, 0], [0, -1]]))
    # Atom A rotates into A, B into B.
    sphi = 2*sympy.pi/3
    RC3 = sympy.Matrix([[sympy.cos(sphi), -sympy.sin(sphi)],
                      [sympy.sin(sphi), sympy.cos(sphi)]])
    C3 = PointGroupElement(RC3, False, False, np.eye(2))

    # Generate graphene Hamiltonian in Kwant from qsymm
    symmetries = [C, TR, C3]
    # Generate using a family
    family = bloch_family(hopping_vectors, symmetries, norbs)
    syst_from_family = bloch_family_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    # Generate using a single Model object
    g = sympy.Symbol('g', real=True)
    ham = hamiltonian_from_family(family, coeffs=[g])
    ham = Model(hamiltonian=ham, momenta=family[0].momenta)
    syst_from_model = bloch_model_to_builder(ham, norbs, lat_vecs, atom_coords)
    
    # Make the graphene Hamiltonian using kwant only
    atoms, orbs = zip(*[(atom, norb) for atom, norb in
                        norbs.items()])
    # Make the kwant lattice
    lat = kwant.lattice.general(lat_vecs,
                                atom_coords,
                                norbs=orbs)
    # Store sublattices by name
    sublattices = {atom: sublat for atom, sublat in
                   zip(atoms, lat.sublattices)}

    sym = kwant.TranslationalSymmetry(*lat_vecs)
    bulk = kwant.Builder(sym)
    
    bulk[ [sublattices['A'](0, 0), sublattices['B'](0, 0)] ] = 0
    
    def hop(site1, site2, c0):
        return c0
    
    bulk[lat.neighbors()] = hop
    
    fsyst_family = kwant.wraparound.wraparound(syst_from_family).finalized()
    fsyst_model = kwant.wraparound.wraparound(syst_from_model).finalized()
    fsyst_kwant = kwant.wraparound.wraparound(bulk).finalized()
    
    # Check that the energies are identical at random points in the Brillouin zone
    coeff = 0.5 + np.random.rand()
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeff, k_x=kx, k_y=ky)
        hamiltonian = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        Es1 = np.linalg.eigh(hamiltonian)[0]
        hamiltonian = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        Es2 = np.linalg.eigh(hamiltonian)[0]
        assert np.allclose(Es1, Es2)
        params = dict(g=coeff, k_x=kx, k_y=ky)
        hamiltonian = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        Es3 = np.linalg.eigh(hamiltonian)[0]
        assert np.allclose(Es2, Es3)
        
    # Include random onsites as well
    one = sympy.numbers.One()
    onsites = [Model({one: np.array([[1, 0], [0, 0]])}, momenta=family[0].momenta),
               Model({one: np.array([[0, 0], [0, 1]])}, momenta=family[0].momenta)]
    family = family + onsites
    syst_from_family = bloch_family_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    gs = list(sympy.symbols('g0:%d'%3, real=True))
    ham = hamiltonian_from_family(family, coeffs=gs)
    ham = Model(hamiltonian=ham, momenta=family[0].momenta)
    syst_from_model = bloch_model_to_builder(ham, norbs, lat_vecs, atom_coords)
    
    def onsite_A(site, c1):
        return c1
    
    def onsite_B(site, c2):
        return c2
    
    bulk[[sublattices['A'](0, 0)]] = onsite_A
    bulk[[sublattices['B'](0, 0)]] = onsite_B
    
    fsyst_family = kwant.wraparound.wraparound(syst_from_family).finalized()
    fsyst_model = kwant.wraparound.wraparound(syst_from_model).finalized()
    fsyst_kwant = kwant.wraparound.wraparound(bulk).finalized()
    
    # Check that the energies are identical at random points in the Brillouin zone
    coeffs = 0.5 + np.random.rand(3)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], c1=coeffs[1], c2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        Es1 = np.linalg.eigh(hamiltonian)[0]
        hamiltonian = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        Es2 = np.linalg.eigh(hamiltonian)[0]
        assert np.allclose(Es1, Es2)
        params = dict(g0=coeffs[0], g1=coeffs[1], g2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        Es3 = np.linalg.eigh(hamiltonian)[0]
        assert np.allclose(Es2, Es3)
