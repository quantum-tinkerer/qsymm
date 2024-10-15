.. _tutorial_bloch_generator:

Generating tight-binding models
===============================

.. seealso::
    The complete source code of this example can be found in :jupyter-download-script:`bloch_generator`.
    A Jupyter notebook can be found in :jupyter-download-notebook:`bloch_generator`.

.. jupyter-kernel::
    :id: bloch_generator

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import sympy

    import qsymm

In addition to finding the symmetry group of a given Hamiltonian, Qsymm can also generate
a class of models that have a given symmetry. In this tutorial we will work through a
few pertinent examples to show how we can use Qsymm to generate a tight-binding model
from symmetry constraints.


Graphene
--------
First we're going to generate the spinless nearest-neighbor tight-binding Hamiltonian
for graphene.


The generators of the symmetry group are time-reversal symmetry, sublattice (or chiral) symmetry,
and threefold rotation symmetry.

.. jupyter-execute::

    # Time reversal
    TR = qsymm.time_reversal(2, U=np.eye(2))

    # Chiral symmetry
    C = qsymm.chiral(2, U=np.array([[1, 0], [0, -1]]))

    # Atom A rotates into A, B into B.
    # We use sympy to get an exact representation of the multiplying factors
    sphi = (2 / 3) * sympy.pi

    C3 = qsymm.PointGroupElement(
        sympy.Matrix([
            [sympy.cos(sphi), -sympy.sin(sphi)],
            [sympy.sin(sphi), sympy.cos(sphi)]
        ]),
        U=np.eye(2),
    )

    symmetries = [C, TR, C3]

There are two carbon atoms per unit cell (A and B) with one orbital each.
The lattice is triangular, and only includes hoppings between nearest neighbour atoms.
This restricts hoppings to only those between atoms of different types, such that each
atom couples to three neighbouring atoms.

Using the symmetrization strategy to generate the Hamiltonian, it is sufficient to specify
hoppings to one such neighbour along with the symmetry generators,
and we take the vector :math:`(1,0)` to connect this neighbouring pair of atoms.

.. jupyter-execute::

    norbs = [('A', 1), ('B', 1)]  # A and B atom per unit cell, one orbital each
    hopping_vectors = [('A', 'B', [0, 1])]  # Hopping between neighbouring A and B atoms

Now we generate the hamiltonian using `~qsymm.hamiltonian_generator.bloch_family`:

.. jupyter-execute::

    family = qsymm.bloch_family(hopping_vectors, symmetries, norbs)
    qsymm.display_family(family)

Scale the bond length in terms of the graphene lattice constant, and have the function
return a list of BlochModel objects. For this we can use a more convenient
definition of the rotation

.. jupyter-execute::

    C3 = qsymm.rotation(1/3, U=np.eye(2))
    symmetries = [C, TR, C3]

    norbs = [('A', 1), ('B', 1)]
    hopping_vectors = [('A', 'B', [0, 1/np.sqrt(3)])]

    family = qsymm.bloch_family(hopping_vectors, symmetries, norbs, bloch_model=True)
    qsymm.display_family(family)


Three-orbital tight-binding model for monolayer :math:`MX_2`
------------------------------------------------------------

We use the Hamiltonian generator to reproduce the tight binding model for
monolayer :math:`MX_2` published in
`Phys. Rev. B 88, 085433 (2013) <https://doi.org/10.1103/PhysRevB.88.085433>`_.

The generators of the symmetry group of the tight binding model are time reversal symmetry,
mirror symmetry and threefold rotation symmetry.

.. jupyter-execute::

    # Time reversal
    TR = qsymm.time_reversal(2, np.eye(3))

    # Mirror symmetry
    Mx = qsymm.mirror([1, 0], np.diag([1, -1, 1]))

    # Threefold rotation on d_z^2, d_xy, d_x^2-y^2 states.
    C3U = np.array([
        [1, 0, 0],
        [0, -0.5, -np.sqrt(3)/2],
        [0, np.sqrt(3)/2, -0.5]
    ])

    # Could also use the predefined representation of rotations on d-orbitals
    Ld = qsymm.groups.L_matrices(3, 2)
    C3U2 = qsymm.groups.spin_rotation(2 * np.pi * np.array([0, 0, 1/3]), Ld)

    # Restrict to d_z^2, d_xy, d_x^2-y^2 states
    mask = np.array([1, 2 ,0])
    C3U2 = C3U2[mask][:, mask]

    assert np.allclose(C3U, C3U2)

    C3 = qsymm.rotation(1/3, U=C3U)

    symmetries = [TR, Mx, C3]

Next, we specify the hoppings to include. The tight binding model has a triangular lattice,
three orbitals per M atom, and nearest neighbour hopping.

.. jupyter-execute::

    # One site per unit cell (M atom), with three orbitals
    norbs = [('a', 3)]

Each atom has six nearest neighbour atoms at a distance of one primitive lattice vector.
Since we use the symmetrization strategy to generate the Hamiltonian, it is sufficient
to specify a hopping to one nearest neighbour atom along with the symmetry generators.
We take the primitive vector connecting the pair of atoms to be :math:`(1,0)`.

.. jupyter-execute::

    # Hopping to a neighbouring atom one primitive lattice vector away
    hopping_vectors = [('a', 'a', [1, 0])]

We again use `~qsymm.hamiltonian_generator.bloch_family` to generate the tight-binding
Hamiltonian:

.. jupyter-execute::

    family = qsymm.bloch_family(hopping_vectors, symmetries, norbs, bloch_model=True)

The Hamiltonian family should include 8 linearly independent components, including the onsite terms.

.. jupyter-execute::

    len(family)

.. jupyter-execute::

    qsymm.display_family(family)


4-site model for monolayer :math:`WTe_2`
----------------------------------------

We use the Hamiltonian generator to reproduce the tight binding model for monolayer WTe2
published in `Phys. Rev. X 6, 041069 (2016) <https://doi.org/10.1103/PhysRevX.6.031021>`_.

The generators of the symmetry group of the tight binding model are time reversal symmetry,
glide reflection and inversion symmetry.

.. jupyter-execute::

    # Define 4 sites with one orbital each
    sites = ['Ad', 'Ap', 'Bd', 'Bp']
    norbs = [(site, 1) for site in sites]

    # Define symbolic coordinates for orbitals
    rAp = qsymm.sympify('[x_Ap, y_Ap]')
    rAd = qsymm.sympify('[x_Ad, y_Ad]')
    rBp = qsymm.sympify('[x_Bp, y_Bp]')
    rBd = qsymm.sympify('[x_Bd, y_Bd]')

    # Define hoppings to include
    hopping_vectors = [
        ('Bd', 'Bd', np.array([1, 0])),
        ('Ap', 'Ap', np.array([1, 0])),
        ('Bd', 'Ap', rAp - rBd),
        ('Ap', 'Bp', rBp - rAp),
        ('Ad', 'Bd', rBd - rAd),
    ]

.. jupyter-execute::

    # Inversion
    perm_inv = {'Ad': 'Bd', 'Ap': 'Bp', 'Bd': 'Ad', 'Bp': 'Ap'}
    onsite_inv = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
    inversion = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv)

    # Glide
    perm_glide = {site: site for site in sites}
    onsite_glide = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
    glide = qsymm.groups.symmetry_from_permutation(np.array([[-1, 0],[0, 1]]), perm_glide, norbs, onsite_glide)

    # TR
    time_reversal = qsymm.time_reversal(2, np.eye(4))

    generators = {glide, inversion, time_reversal}
    sg = qsymm.groups.generate_group(generators)

Again we generate the tight-binding Hamiltonian:

.. jupyter-execute::

    family = qsymm.bloch_family(hopping_vectors, generators, norbs=norbs)
    qsymm.display_family(family)


Square lattice with 4 sites in the unit cell
--------------------------------------------

Now we're going to make a model with square lattice that has 4 sites in the unit cell
related by 4-fold rotation. Sites have spin-1/2 and we add time reversal and particle-hole symmetry.

.. jupyter-execute::

    hopping_vectors = [
        ('a', 'b', np.array([1, 0])),
        ('b', 'a', np.array([1, 0])),
        ('c', 'd', np.array([1, 0])),
        ('d', 'c', np.array([1, 0])),
        ('a', 'c', np.array([0, 1])),
        ('c', 'a', np.array([0, 1])),
        ('b', 'd', np.array([0, 1])),
        ('d', 'b', np.array([0, 1])),
    ]

    # Define spin-1/2 operators
    S = qsymm.groups.spin_matrices(1/2)
    # Define real space rotation generators in 2D
    L = qsymm.groups.L_matrices(d=2)

    sites = ['a', 'b', 'c', 'd']
    norbs = [(site, 2) for site in sites]

    perm_C4 = {'a': 'b', 'b': 'd', 'd': 'c', 'c': 'a'}
    onsite_C4 = {site: qsymm.groups.spin_rotation(2*np.pi * np.array([0, 0, 1/4]), S) for site in sites}
    C4 = qsymm.groups.symmetry_from_permutation(
        qsymm.groups.spin_rotation(2*np.pi * np.array([1/4]), L, roundint=True),
        perm_C4,
        norbs,
        onsite_C4,
    )

    # Fermionic time-reversal
    time_reversal = qsymm.time_reversal(
        2,
        np.kron(np.eye(4),
        qsymm.groups.spin_rotation(2*np.pi * np.array([0, 1/2, 0]), S)),
    )

    # define strange PH symmetry
    particle_hole = qsymm.particle_hole(2, np.eye(8))

    generators = {C4, time_reversal, particle_hole}
    sg = qsymm.groups.generate_group(generators)

Once again we generate the tight-binding Hamiltonian:

.. jupyter-execute::

    family = qsymm.bloch_family(hopping_vectors, generators, norbs=norbs)
    qsymm.display_family(family)
