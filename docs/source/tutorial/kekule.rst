.. _tutorial_kekule:

Finding symmetries of the Kekule-Y continuum model
==================================================

.. seealso::
    The complete source code of this example can be found in :jupyter-download:script:`kekule`.
    A Jupyter notebook can be found in :jupyter-download:notebook:`kekule`.

.. jupyter-kernel::
    :id: kekule

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import sympy

    import qsymm


Kekule-Y
--------

Find symmetries of Kekule-Y effective model, starting from the full hexagonal symmetry group
with time reversal and particle-hole symmetry.

.. jupyter-execute::

    ham_kek_y = """
        vs * kron((k_x * sigma_x + k_y * sigma_y), eye(2))
        + vt * kron(eye(2), (k_x * sigma_x + k_y * sigma_y))
    """
    display(qsymm.sympify(ham_kek_y))

.. jupyter-execute::

    H_kek_y= qsymm.Model(ham_kek_y, momenta=['k_x', 'k_y'])
    candidates = qsymm.groups.hexagonal()
    sgy, cg = qsymm.symmetries(H_kek_y, candidates)

Enumerate subgroups that protect double Dirac cone:

.. jupyter-execute::

    all_subgroups = qsymm.groups.generate_subgroups(sgy)

.. jupyter-execute::

    # Terms proportional to the identity matrix don't open a gap.
    identity_terms = [qsymm.Model({qsymm.sympify(1): np.eye(4)})]

    dd_sg = []
    for subgroup, gens in all_subgroups.items():
        family = qsymm.continuum_hamiltonian(gens, 2, 0)
        family = qsymm.hamiltonian_generator.subtract_family(family, identity_terms)
        if family == []:
            dd_sg.append(subgroup)

    dd_sg.sort(key=len)

    dd_sg[0]


Twofold rotation combined with sublattice symmetry is enough.
However 2-fold rotation is not a physical symmetry of the lattice model.
Let's get rid of 2-fold rotations:

.. jupyter-execute::

    rdd_sg = []
    for subgroup, gens in all_subgroups.items():
        invs = {
            qsymm.inversion(2) * g
            for g in [
                qsymm.identity(2),
                qsymm.time_reversal(2),
                qsymm.particle_hole(2),
                qsymm.chiral(2),
            ]
        }
        # Skip subgroups that have inversion
        if subgroup & invs:
            continue
        family = qsymm.continuum_hamiltonian(gens, 2, 0)
        family = qsymm.hamiltonian_generator.subtract_family(family, identity_terms)
        if family == []:
            rdd_sg.append(subgroup)

    rdd_sg.sort(key=len)

    rdd_sg[0]


Sublattice symmetry with 3-fold rotations is the minimal symmetry group required.
Let's test if any of the solutions has no sublattice symmetry:

.. jupyter-execute::

    [sg for sg in rdd_sg if not any(g.antisymmetry for g in sg)]

Sublattice symmetry is required for the protection of the double Dirac cone.


Continuous rotation symmetries of continuum models
--------------------------------------------------

Kekule-Y
********

.. jupyter-execute::

    _, cg = qsymm.symmetries(H_kek_y, candidates=[], continuous_rotations=True, prettify=True)
    cg

Kekule-O
********

.. jupyter-execute::

    ham_kek_o= """
        vs * (
            k_x * kron(eye(2), sigma_x)
            + k_y * kron(sigma_z, sigma_y)
        )
        + Delta * kron(sigma_y, sigma_y)
    """
    display(qsymm.sympify(ham_kek_o))

.. jupyter-execute::

    H_kek_o= qsymm.Model(ham_kek_o, momenta=['k_x', 'k_y'])
    _, cg = qsymm.symmetries(H_kek_o, candidates=[], continuous_rotations=True, prettify=True)
    cg

The rotation generators are different in the two cases


Lattice models
--------------

Generate Kekule-O model
***********************

6 sites (3 A and 3 B) per unit cell, spinless orbitals.
Time-reversal, sublattice and 6-fold rotation and mirror symmetry.

.. jupyter-execute::

    sites = ['A1', 'B1', 'A2', 'B2', 'A3', 'B3']
    norbs = [(site, 1) for site in sites]

    # Time reversal
    TR = qsymm.PointGroupElement(sympy.eye(2), True, False, np.eye(6))

    # Chiral symmetry
    C = qsymm.PointGroupElement(sympy.eye(2), False, True, np.kron(np.eye(3), ([[1, 0], [0, -1]])))

    # Atom A rotates into B, B into A.
    sphi = 2*sympy.pi/6
    RC6 = sympy.Matrix([[sympy.cos(sphi), -sympy.sin(sphi)],
                      [sympy.sin(sphi), sympy.cos(sphi)]])
    permC6 = {'A1': 'B1', 'B1': 'A2', 'A2': 'B2', 'B2': 'A3', 'A3': 'B3', 'B3': 'A1'}
    C6 = qsymm.groups.symmetry_from_permutation(RC6, permC6, norbs)

    RMx = sympy.Matrix([[-1, 0], [0, 1]])
    permMx = {'A1': 'A3', 'B1': 'B2', 'A2': 'A2', 'B2': 'B1', 'A3': 'A1', 'B3': 'B3'}
    Mx = qsymm.groups.symmetry_from_permutation(RMx, permMx, norbs)

    symmetries = [C, TR, C6, Mx]

.. jupyter-execute::

    # Only need the unique hoppings, one weak and one strong
    hopping_vectors = [
        ('A1', 'B1', np.array([0, -1])), # around ring
        ('A2', 'B3', np.array([0, -1])), # next UC
    ]

.. jupyter-execute::

    # construct Hamiltonian terms to symmetry-allowed terms
    family = qsymm.bloch_family(hopping_vectors, symmetries, norbs=norbs)
    qsymm.display_family(family)

Make terms at k=0 and check degeneracy for some linear combination.

.. jupyter-execute::

    Gamma_terms = [term.subs(dict(k_x=0, k_y=0)) for term in family]
    evals, evecs = np.linalg.eigh(
        np.sum([
            list(term.values())[0] * coeff
            for term, coeff in zip(Gamma_terms, [1, 2])
        ],
        axis=0)
    )

    evals

Find symmetry representation in the low energy subspace

.. jupyter-execute::

    # projector to low energy subspace
    proj = evecs[:, 1:5]
    UC3 = (C6 * C6).U

    # projected symmetry operator
    pC3 = proj.T.conj() @ UC3 @ proj

    assert np.allclose(pC3 @ pC3.T.conj(), np.eye(4))

    evals2, evecs2 = np.linalg.eig(pC3)

    # Check angles of the eigenvalues
    np.angle(evals2) / (2 * np.pi)



Generate Kekule-Y model
***********************

6 sites (3 A and 3 B) per unit cell, spinless orbitals.
Time-reversal, sublattice and 3-fold rotation and mirror symmetry.

.. jupyter-execute::

    sites = ['A1', 'B1', 'A2', 'B2', 'A3', 'B3']
    norbs = [(site, 1) for site in sites]

    # Time reversal
    TR = qsymm.time_reversal(2, U=np.eye(6))

    # Chiral symmetry
    C = qsymm.chiral(2, U=np.kron(np.eye(3), ([[1, 0], [0, -1]])))

    # Atom A rotates into B, B into A.
    sphi = 2*sympy.pi/3
    RC3 = sympy.Matrix([
        [sympy.cos(sphi), -sympy.sin(sphi)],
        [sympy.sin(sphi), sympy.cos(sphi)],
    ])
    permC3 = {'A1': 'A1', 'B1': 'B3', 'A2': 'A2', 'B2': 'B1', 'A3': 'A3', 'B3': 'B2'}
    C3 = qsymm.groups.symmetry_from_permutation(RC3, permC3, norbs)

    RMx = sympy.Matrix([[-1, 0], [0, 1]])
    permMx = {'A1': 'A1', 'B1': 'B1', 'A2': 'A3', 'B2': 'B3', 'A3': 'A2', 'B3': 'B2'}
    Mx = qsymm.groups.symmetry_from_permutation(RMx, permMx, norbs)

    symmetries = [C, TR, C3, Mx]

.. jupyter-execute::

    # Only need the unique hoppings, one weak and one strong
    hopping_vectors = [
        ('A1', 'B1', np.array([0, -1])), # Y
        ('A2', 'B3', np.array([0, -1])), # other
    ]

.. jupyter-execute::

    # construct Hamiltonian terms to symmetry-allowed terms
    family = qsymm.bloch_family(hopping_vectors, symmetries, norbs=norbs)
    qsymm.display_family(family)

Make terms at k=0 and check degeneracy for some linear combination.

.. jupyter-execute::

    Gamma_terms = [term.subs(dict(k_x=0, k_y=0)) for term in family]
    evals, evecs = np.linalg.eigh(
        np.sum([
            list(term.values())[0] * coeff
            for term, coeff in zip(Gamma_terms, [1, 2])
        ],
        axis=0)
    )

    evals

Find symmetry representation of 3-fold rotations in the low energy subspace

.. jupyter-execute::

    # projector to low energy subspace
    proj = evecs[:, 1:5]
    UC3 = C3.U

    # projected symmetry operator
    pC3 = proj.T.conj() @ UC3 @ proj

    assert np.allclose(pC3 @ pC3.T.conj(), np.eye(4))

    evals2, evecs2 = np.linalg.eig(pC3)

    # Check angles of the eigenvalues
    np.angle(evals2) / (2 * np.pi)

The symmetry representation in the low energy subspace near the Gamma point is different in the two cases.
