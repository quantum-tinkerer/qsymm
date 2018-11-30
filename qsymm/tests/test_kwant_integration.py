import pytest
import numpy as np
import sympy
import kwant
import warnings
from collections import OrderedDict

from ..symmetry_finder import symmetries
from ..hamiltonian_generator import bloch_family, hamiltonian_from_family
from ..groups import hexagonal, PointGroupElement, spin_matrices, spin_rotation, \
                     ContinuousGroupGenerator
from ..model import Model, e, I, _commutative_momenta
from ..kwant_integration import builder_to_model, bloch_to_builder, builder_discrete_symmetries
from ..linalg import allclose


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
    syst = kwant.Builder(symmetry=kwant.lattice.TranslationalSymmetry(*lat.prim_vecs))
    syst[lat.a(0, 0)] = 1
    syst[lat.b(0, 0)] = 1
    syst[lat.neighbors(1)] = -1

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 24
    assert len(cs) == 0

    # Test simple honeycomb model with value functions
    syst = kwant.Builder(symmetry=kwant.lattice.TranslationalSymmetry(*lat.prim_vecs))
    syst[lat.a(0, 0)] = lambda site, ma: ma
    syst[lat.b(0, 0)] = lambda site, mb: mb
    syst[lat.neighbors(1)] = lambda site1, site2, t: t

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 12
    assert len(cs) == 0
    
    
def test_builder_discrete_symmetries():
    syst = kwant.Builder(particle_hole=np.eye(2),
                         conservation_law=2*np.eye(2))
    builder_symmetries = builder_discrete_symmetries(syst, spatial_dimensions=2)
    
    assert len(builder_symmetries) == 2
    P = builder_symmetries['particle_hole']
    assert isinstance(P, PointGroupElement)
    assert allclose(P.U, np.eye(2))
    assert P.conjugate and P.antisymmetry
    assert allclose(P.R, np.eye(2))

    cons = builder_symmetries['conservation_law']
    assert isinstance(cons, ContinuousGroupGenerator)
    assert allclose(cons.U, 2*np.eye(2))
    assert cons.R is None
    
    syst = kwant.Builder()
    builder_symmetries = builder_discrete_symmetries(syst)
    assert len(builder_symmetries) == 0

    
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

    H = builder_to_model(syst)
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

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False), prettify=True)
    assert len(sg) == 24
    assert len(cs) == 0


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
    syst_from_family = bloch_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    # Generate using a single Model object
    g = sympy.Symbol('g', real=True)
    ham = hamiltonian_from_family(family, coeffs=[g])
    ham = Model(hamiltonian=ham, momenta=family[0].momenta)
    syst_from_model = bloch_to_builder(ham, norbs, lat_vecs, atom_coords)
    
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
        hamiltonian1 = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        hamiltonian2 = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian1, hamiltonian2)
        params = dict(g=coeff, k_x=kx, k_y=ky)
        hamiltonian3 = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian2, hamiltonian3)
        
    # Include random onsites as well
    one = sympy.numbers.One()
    onsites = [Model({one: np.array([[1, 0], [0, 0]])}, momenta=family[0].momenta),
               Model({one: np.array([[0, 0], [0, 1]])}, momenta=family[0].momenta)]
    family = family + onsites
    syst_from_family = bloch_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    gs = list(sympy.symbols('g0:%d'%3, real=True))
    ham = hamiltonian_from_family(family, coeffs=gs)
    ham = Model(hamiltonian=ham, momenta=family[0].momenta)
    syst_from_model = bloch_to_builder(ham, norbs, lat_vecs, atom_coords)
    
    def onsite_A(site, c1):
        return c1
    
    def onsite_B(site, c2):
        return c2
    
    bulk[[sublattices['A'](0, 0)]] = onsite_A
    bulk[[sublattices['B'](0, 0)]] = onsite_B
    
    fsyst_family = kwant.wraparound.wraparound(syst_from_family).finalized()
    fsyst_model = kwant.wraparound.wraparound(syst_from_model).finalized()
    fsyst_kwant = kwant.wraparound.wraparound(bulk).finalized()
    
    # Check equivalence of the Hamiltonian at random points in the BZ
    coeffs = 0.5 + np.random.rand(3)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], c1=coeffs[1], c2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian1 = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        hamiltonian2 = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian1, hamiltonian2)
        params = dict(g0=coeffs[0], g1=coeffs[1], g2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian3 = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian2, hamiltonian3)


def test_wraparound_convention():
    # Test that it matches exactly kwant.wraparound convention
    # Make the graphene Hamiltonian using kwant only
    norbs = OrderedDict({'A': 1, 'B': 1})  # A and B atom per unit cell, one orbital each
    atoms, orbs = zip(*[(atom, norb) for atom, norb in
                        norbs.items()])
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0), (1, 0)]
    # We set the interatom distance to 1, so the lattice vectors have length sqrt(3)
    lat_vecs = [(3/2, np.sqrt(3)/2), (3/2, -np.sqrt(3)/2)]
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

    wrapped = kwant.wraparound.wraparound(bulk).finalized()
    ham2 = builder_to_model(bulk, unit_cell_convention=True)
    # Check that the Hamiltonians are identical at random points in the Brillouin zone
    H1 = wrapped.hamiltonian_submatrix
    H2 = ham2.lambdify()
    coeffs = 0.5 + np.random.rand(1)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], k_x=kx, k_y=ky)
        h1, h2 = H1(params=params), H2(**params)
        assert allclose(h1, h2), (h1, h2)



def test_inverse_transform():
    # Define family on square lattice
    s = spin_matrices(1/2)
    # Time reversal
    TR = PointGroupElement(np.eye(2), True, False,
                           spin_rotation(2 * np.pi * np.array([0, 1/2, 0]), s))
    # Mirror symmetry
    Mx = PointGroupElement(np.array([[-1, 0], [0, 1]]), False, False,
                           spin_rotation(2 * np.pi * np.array([1/2, 0, 0]), s))
    # Fourfold
    C4 = PointGroupElement(np.array([[0, 1], [-1, 0]]), False, False,
                           spin_rotation(2 * np.pi * np.array([0, 0, 1/4]), s))
    symmetries = [TR, Mx, C4]

    # One site per unit cell
    norbs = OrderedDict({'A': 2})
    # Hopping to a neighbouring atom one primitive lattice vector away
    hopping_vectors = [('A', 'A', [1, 0])]
    # Make family
    family = bloch_family(hopping_vectors, symmetries, norbs)
    fam = hamiltonian_from_family(family, tosympy=False)
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0)]
    lat_vecs = [(1, 0), (0, 1)]
    syst = bloch_to_builder(fam, norbs, lat_vecs, atom_coords)
    # Convert it back
    ham2 = builder_to_model(syst).tomodel(nsimplify=True)
    # Check that it's the same as the original
    assert fam == ham2

    # Check that the Hamiltonians are identical at random points in the Brillouin zone
    sysw = kwant.wraparound.wraparound(syst).finalized()
    H1 = sysw.hamiltonian_submatrix
    H2 = ham2.lambdify()
    H3 = fam.lambdify()
    coeffs = 0.5 + np.random.rand(3)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], c1=coeffs[1], c2=coeffs[2], k_x=kx, k_y=ky)
        assert allclose(H1(params=params), H2(**params))
        assert allclose(H1(params=params), H3(**params))

        
def test_consistency_kwant():
    """Make a random 1D Model, convert it to a builder, and compare
    the Bloch representation of the Model with that which Kwant uses
    in wraparound and in Bands. Then, convert the builder back to a Model
    and compare with the original Model.
    For comparison, we also make the system using Kwant only.
    """
    orbs = 4
    T = np.random.rand(2*orbs, 2*orbs) + 1j*np.random.rand(2*orbs, 2*orbs)
    H = np.random.rand(2*orbs, 2*orbs) + 1j*np.random.rand(2*orbs, 2*orbs)
    H += H.T.conj()

    # Make the 1D Model manually using only qsymm features.
    c0, c1 = sympy.symbols('c0 c1', real=True)
    kx = _commutative_momenta[0]
    
    Ham = Model({c0 * e**(-I*kx): T}, momenta=[0])
    Ham += Ham.T().conj()
    Ham += Model({c1: H}, momenta=[0]) 

    # Two superimposed atoms, same number of orbitals on each
    norbs = OrderedDict({'A': orbs, 'B': orbs}) 
    atom_coords = [(0.3, ), (0.3, )]
    lat_vecs = [(1, )] # Lattice vector
    
    # Make a Kwant builder out of the qsymm Model
    model_syst = bloch_to_builder(Ham, norbs, lat_vecs, atom_coords)
    fmodel_syst = model_syst.finalized()
    
    # Make the same system manually using only Kwant features.
    lat = kwant.lattice.general(np.array([[1.]]),
                            [(0., )],
                            norbs=2*orbs)
    kwant_syst = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))

    def onsite(site, c1):
        return c1*H

    def hopping(site1, site2, c0):
        return c0*T
    
    sublat = lat.sublattices[0]
    kwant_syst[sublat(0,)] = onsite
    hopp = kwant.builder.HoppingKind((1, ), sublat)
    kwant_syst[hopp] = hopping
    fkwant_syst = kwant_syst.finalized()
    
    # Make sure we are consistent with bands calculations in kwant
    # The Bloch Hamiltonian used in Kwant for the bands computation
    # is h(k) = exp(-i*k)*hop + onsite + exp(i*k)*hop.T.conj.
    # We also check that all is consistent with wraparound
    coeffs = (0.7, 1.2)
    params = dict(c0 = coeffs[0], c1 = coeffs[1])
    kwant_hop = fkwant_syst.inter_cell_hopping(params=params)
    kwant_onsite = fkwant_syst.cell_hamiltonian(params=params)
    model_kwant_hop = fmodel_syst.inter_cell_hopping(params=params)
    model_kwant_onsite = fmodel_syst.cell_hamiltonian(params=params)
    
    assert allclose(model_kwant_hop, coeffs[0]*T)
    assert allclose(model_kwant_hop, kwant_hop)
    assert allclose(model_kwant_onsite, kwant_onsite)
    
    h_model_kwant = (lambda k: np.exp(-1j*k)*model_kwant_hop + model_kwant_onsite +
                     np.exp(1j*k)*model_kwant_hop.T.conj()) # As in kwant.Bands
    h_model = Ham.lambdify()
    wsyst = kwant.wraparound.wraparound(model_syst).finalized()
    for _ in range(20):
        k = (np.random.rand() - 0.5)*2*np.pi
        assert allclose(h_model_kwant(k), h_model(coeffs[0], coeffs[1], k))
        params['k_x'] = k
        h_wrap = wsyst.hamiltonian_submatrix(params=params)
        assert allclose(h_model(coeffs[0], coeffs[1], k), h_wrap)

    # Get the model back from the builder
    # From the Kwant builder based on original Model
    Ham1 = builder_to_model(model_syst, momenta=Ham.momenta).tomodel(nsimplify=True)
    # From the pure Kwant builder
    Ham2 = builder_to_model(kwant_syst, momenta=Ham.momenta).tomodel(nsimplify=True)
    assert Ham == Ham1
    assert Ham == Ham2
