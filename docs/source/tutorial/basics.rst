.. _tutorial_basics:

Qsymm Basics
============


.. seealso::
    The complete source code of this example can be found in :jupyter-download-script:`basics`.
    A Jupyter notebook can be found in :jupyter-download-notebook:`basics`.

.. jupyter-kernel::
    :id: basics

Getting started with Qsymm is as simple as importing it:

.. jupyter-execute::

    import qsymm

To make effective use of Qsymm we'll also need a few other utilities: ``numpy``
handling numeric arrays, and ``sympy`` for symbolic mathematics:

.. jupyter-execute::

    import numpy as np
    import sympy

In all the following tutorials we will use these standard imports, and they won't
be explicitly shown


Defining a Qsymm model
----------------------

Let's start by defining a 3D Rashba Hamiltonian symbolically as a Python string:

.. jupyter-execute::

    ham = ("hbar^2 / (2 * m) * (k_x**2 + k_y**2 + k_z**2) * eye(2) +" +
            "alpha * sigma_x * k_x + alpha * sigma_y * k_y + alpha * sigma_z * k_z")

We can then create a Qsymm `~qsymm.model.Model` directly from this symbolic Hamiltonian:

.. jupyter-execute::

    H = qsymm.Model(ham)

We can then directly inspect the contents by printing the `~qsymm.model.Model`:

.. jupyter-execute::

    print(H)

We can also extract a more readable representation by using the ``tosympy`` method, which
converts the `~qsymm.model.Model` to a ``sympy`` expression:

.. jupyter-execute::

    H.tosympy(nsimplify=True)

The argument ``nsimplify=True`` makes the output more readable by forcing ``sympy`` to elide
factors of ``1.0`` that multiply each term. Note that Qsymm automatically interprets the symbols
``sigma_x``, ``sigma_y`` and ``sigma_z`` as the Pauli matrices, and ``eye(2)`` as the 2x2
identity matrix.

`~qsymm.model.Model` as a ``momenta`` attribute that specifies which symbols are considered the
momentum variables:

.. jupyter-execute::

    H.momenta

By default Qsymm assumes that your model is written in 3D (even if it does not include all 3
momenta). To define a lower-dimensional model you must explicitly specify the momentum
variables, e.g:

.. jupyter-execute::

    ham2D = ("hbar^2 / (2 * m) * (k_x**2 + k_z**2) * eye(2) +" +
             "alpha * sigma_x * k_x + alpha * sigma_y * k_z")
    H2D = qsymm.Model(ham2D, momenta=['k_x', 'k_z'])

.. jupyter-execute::

    H2D.tosympy(nsimplify=True)

.. jupyter-execute::

    H2D.momenta


Defining group elements
-----------------------
Qsymm is all about finding and generating symmetries of models, so it is unsurprising
that it contains utilities for defining group elements.

Below are a few examples of the sorts of things you can define with Qsymm:

.. jupyter-execute::

    # Identity in 3D
    E = qsymm.identity(3)
    # Inversion in 3D
    I = qsymm.inversion(3)
    # 4-fold rotation around the x-axis
    C4 = qsymm.rotation(1/4, [1, 0, 0])
    # 3-fold rotation around the [1, 1, 1] axis
    C3 = qsymm.rotation(1/3, [1, 1, 1])
    # Time reversal
    TR = qsymm.time_reversal(3)
    # Particle-hole
    PH = qsymm.particle_hole(3)

The documentation page of the `qsymm.groups` module contains an exhaustive list
of what can be generated.

As with other Qsymm objects we can get a readable representation of these
group elements:

.. jupyter-execute::

    C4

.. jupyter-execute::

    TR

Given a set of group generators we can also generate a group:

.. jupyter-execute::

    cubic_gens = {I, C4, C3, TR, PH}
    cubic_group = qsymm.groups.generate_group(cubic_gens)
    print(len(cubic_group))

Group elements can be multiplied and inverted, as we would expect:

.. jupyter-execute::

    C3 * C4

.. jupyter-execute::

    C3**-1

We can also apply group elements to the `~qsymm.model.Model` that we defined
in the previous section:

.. jupyter-execute::

    H_with_TR = TR.apply(H)
    H_with_TR.tosympy(nsimplify=True)


Defining continuous group generators
------------------------------------
In addition to the group elements we can also define generators of continuous groups
using `qsymm.groups.ContinuousGroupGenerator`:

.. jupyter-execute::

    sz = qsymm.ContinuousGroupGenerator(None, np.array([[1, 0], [0, -1]]))

The first argument to `~qsymm.groups.ContinuousGroupGenerator` is the realspace rotation generator;
by specifying ``None`` we indicate that we want the rotation part to be zero. The second
argument is the unitary action of the generator on the Hilbert space as a Hermitian matrix.

Applying a `~qsymm.groups.ContinuousGroupGenerator` to a `~qsymm.model.Model` calculates the commutator:

.. jupyter-execute::

    sz.apply(H).tosympy(nsimplify=True)

For the 3D Rashba Hamiltonian we defined at the start of the tutorial spin-z is not conserved, hence
the commutator is non-zero.


Finding symmetries
------------------
The function `~qsymm.symmetry_finder.symmetries` allows us to find the symmetries of a
`~qsymm.model.Model`. Let us find out whether the 3D Rashba Hamiltonian defined earlier
has cubic group symmetry:

.. jupyter-execute::

    discrete_symm, continuous_symm = qsymm.symmetries(H, cubic_group)
    print(len(discrete_symm), len(continuous_symm))

It has 48 discrete symmetries (cubic group without inversion and time-reversal) and
no continuous symmetries (conserved quantities).

For more detailed examples see :ref:`tutorial_symmetry_finder` and :ref:`tutorial_kekule`.


Generating Hamiltonians from symmetry constraints
-------------------------------------------------
The `qsymm.hamiltonian_generator` module contains algorithms for generating Hamiltonians from
symmetry constraints.

For example let us generate all 2-band :math:`k \cdot p` Hamiltonians with the same discrete
symmetries as the Rashba Hamiltonian that we found in the previous section:

.. jupyter-execute::

    family = qsymm.continuum_hamiltonian(discrete_symm, dim=3, total_power=2, prettify=True)
    qsymm.display_family(family)

It is exactly the Hamiltonian family we started with.

For more detailed examples see :ref:`tutorial_kdotp_generator`, :ref:`tutorial_bloch_generator`
and :ref:`tutorial_kekule`.


Saving and loading Qsymm models
-------------------------------------------------
Qsymm models and identified symmetries don't guarantee consistent ordering and basis selection
across multiple runs. To avoid irrerproducible results you may use the ``Model.tosympy`` method
and serialize the resulting sympy expression as shown below.

To save we do:

.. jupyter-execute::

    H2D_sympy = H2D.tosympy()

    with open("H2D.txt", "w") as f:
        f.write(str(H2D))

To load we do:

.. jupyter-execute::

    with open("H2D.txt") as f:
        data = f.read()

    loaded_H2D = qsymm.Model(sympy.parsing.sympy_parser.parse_expr(f), momenta=['k_x', 'k_z'])
