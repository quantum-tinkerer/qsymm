import pytest
import numpy as np
import scipy.linalg as la
from copy import deepcopy
import itertools as it

from ..groups import generate_group, generate_subgroups
from ..symmetry_finder import discrete_symmetries
from ..model import Model
from ..groups import PointGroupElement, spin_matrices, spin_rotation, L_matrices
from ..hamiltonian_generator import continuum_hamiltonian
from ..linalg import prop_to_id
from ..kwant_continuum import sympify

sigma = spin_matrices(1/2)

def test_mutual_continuum():
    # Tests that check continuum hamiltonian generator against symmetry finder

    # 0D
    dim = 0
    for N in np.arange(2, 5):
        # Symplistic symmetry action
        TR = PointGroupElement(np.eye(dim), True, False, np.eye(N))
        PH = PointGroupElement(np.eye(dim), True, True, np.eye(N))
        gens = {TR, PH}
        group = generate_group(gens)
        groupnoU = deepcopy(group)
        for g in groupnoU:
            g.U = None
        subgroups = generate_subgroups(group)
        for sg, gen in subgroups.items():
            families = continuum_hamiltonian(list(gen), dim, 3)
            H = Model({sympify('a_' + str(i)) * k: v
                for i, fam in enumerate(families) for k, v in fam.items()},
                momenta = range(dim))
            if not H == {}:
                sg2, Ps = discrete_symmetries(H, groupnoU)
                # new symmetry group may bigger because of additional constraints
                assert sg2 >= sg, (sg2, sg)
                for g1, g2 in it.product(sg, sg2):
                    if g1 == g2:
                        prop, coeff = prop_to_id((g1.inv() * g2).U)
                        assert prop and np.isclose(abs(coeff), 1)

    # More realistic symmetry action
    TR = PointGroupElement(np.eye(dim), True, False, np.kron(sigma[1], np.eye(2)))
    PH = PointGroupElement(np.eye(dim), True, True, np.kron(np.eye(2), sigma[0]))
    gens = {TR, PH}
    group = generate_group(gens)
    groupnoU = deepcopy(group)
    for g in groupnoU:
        g.U = None
    subgroups = generate_subgroups(group)
    for sg, gen in subgroups.items():
        families = continuum_hamiltonian(list(gen), dim, 3)
        H = Model({sympify('a_' + str(i)) * k: v
            for i, fam in enumerate(families) for k, v in fam.items()},
            momenta = range(dim))
        if not H == {}:
            sg2, Ps = discrete_symmetries(H, groupnoU)
            # new symmetry group may bigger because of additional constraints
            assert sg2 == sg, (sg2, sg)
            for g1, g2 in it.product(sg, sg2):
                if g1 == g2:
                    prop, coeff = prop_to_id((g1.inv() * g2).U)
                    assert prop and np.isclose(abs(coeff), 1)

    # 1D
    dim = 1
    # More realistic symmetry action
    TR = PointGroupElement(np.eye(dim), True, False, np.kron(sigma[1], np.eye(2)))
    PH = PointGroupElement(np.eye(dim), True, True, np.kron(np.eye(2), sigma[2]))
    I = PointGroupElement(-np.eye(dim), False, False, np.eye(4))
    gens = {TR, PH, I}
    group = generate_group(gens)
    groupnoU = deepcopy(group)
    for g in groupnoU:
        g.U = None
    subgroups = generate_subgroups(group)
    for sg, gen in subgroups.items():
        families = continuum_hamiltonian(list(gen), dim, 1)
        H = Model({sympify('a_' + str(i)) * k: v
            for i, fam in enumerate(families) for k, v in fam.items()},
            momenta = range(dim))
        if not H == {}:
            sg2, Ps = discrete_symmetries(H, groupnoU)
            assert sg2 == sg, (sg2, sg)
            for g1, g2 in it.product(sg, sg2):
                if g1 == g2:
                    prop, coeff = prop_to_id((g1.inv() * g2).U)
                    assert prop and np.isclose(abs(coeff), 1)

    # 2D
    dim = 2
    L = L_matrices(2)
    # More realistic symmetry action
    TR = PointGroupElement(np.eye(dim), True, False, np.kron(sigma[1], np.eye(2)))
    PH = PointGroupElement(np.eye(dim), True, True, np.kron(np.eye(2), sigma[0]))
    M = PointGroupElement(-np.array([[-1, 0], [0, 1]]), False, False,
                         la.block_diag(spin_rotation(np.pi*np.array([1, 0, 0]), sigma),
                                        spin_rotation(np.pi*np.array([1, 0, 0]), sigma).conj()))
    n = np.pi * np.array([0, 0, 1/2])
    C4 = PointGroupElement(spin_rotation([n[2]], L, roundint=True), False, False,
                         la.block_diag(spin_rotation(n, sigma),
                                        spin_rotation(n, sigma).conj()))
    gens = {TR, PH, M, C4}
    group = generate_group(gens)
    groupnoU = deepcopy(group)
    for g in groupnoU:
        g.U = None
    families = continuum_hamiltonian(list(gens), dim, 1)
    H = Model({sympify('a_' + str(i)) * k: v
        for i, fam in enumerate(families) for k, v in fam.items()},
        momenta = range(dim))
    if not H == {}:
        sg2, Ps = discrete_symmetries(H, groupnoU)
        assert sg2 == group, (sg2, group)
        for g1, g2 in it.product(group, sg2):
            if g1 == g2:
                prop, coeff = prop_to_id((g1.inv() * g2).U)
                assert prop and np.isclose(abs(coeff), 1)
