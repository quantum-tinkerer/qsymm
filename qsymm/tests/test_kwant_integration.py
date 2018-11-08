import pytest
import numpy as np
import kwant
import warnings

from ..symmetry_finder import symmetries
from ..groups import hexagonal
from ..kwant_integration import builder_to_model, bravais_point_group


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
