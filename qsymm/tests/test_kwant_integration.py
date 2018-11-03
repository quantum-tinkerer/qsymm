import pytest
import kwant

from ..symmetry_finder import symmetries
from ..groups import hexagonal
from ..kwant_integration import builder_to_model


def _version_higher(v='1.4.0'):
    from kwant import __version__ as n
    v = tuple(int(char) for char in  v[:5].split('.'))
    n = tuple(int(char) for char in  n[:5].split('.'))
    if n >= v:
        return True
    return False


def test_honeycomb():
    assert _version_higher(v='1.4.0'), 'needs kwant >= 1.4.0'

    lat = kwant.lattice.honeycomb(norbs=1)

    # Test simple honeycomb model with constant terms
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