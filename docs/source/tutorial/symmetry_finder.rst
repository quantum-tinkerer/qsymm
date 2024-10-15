.. _tutorial_symmetry_finder:

Finding Symmetries
==================


.. seealso::
    The complete source code of this example can be found in :jupyter-download-script:`symmetry_finder`.
    A Jupyter notebook can be found in :jupyter-download-notebook:`symmetry_finder`.

.. jupyter-kernel::
    :id: symmetry_finder

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import sympy
    import qsymm

A basic example
---------------
Let's start from the same 3D Rashba hamiltonian we used in :ref:`the basic tutorial <tutorial_basics>`.

.. jupyter-execute::

    ham = ("hbar^2 / (2 * m) * (k_x**2 + k_y**2 + k_z**2) * eye(2) +" +
            "alpha * sigma_x * k_x + alpha * sigma_y * k_y + alpha * sigma_z * k_z")
    # Convert to standard monomials form
    H = qsymm.Model(ham)
    H.tosympy()

We start from the cubic group as the set of candidates for point group symmetries:

.. jupyter-execute::

    cubic_group = qsymm.groups.cubic()

Then we can use `~qsymm.symmetry_finder.symmetries` to find point group symmetries,
discrete onsite symmetries, and continuous symmetries:

.. jupyter-execute::

    sg, cg = qsymm.symmetries(H, cubic_group)

As before we see that we have 48 symmetries, but no continuous symmetries:

.. jupyter-execute::

    print(len(sg), len(cg))

If we take a look at a few of the symmetries, we see that it includes time reversal
but not inversion:

.. jupyter-execute::

    from IPython.display import display

    for i in range(1, 5):
      display(sg[-i])

We can also get a more detailed representation of these group elements:

.. jupyter-execute::

    from IPython.display import Math

    display(Math(qsymm.groups.pretty_print_pge(sg[28], latex=True, full=True)))

We can check that this is the same as the full cubic group without inversion, plus time-reversal


.. jupyter-execute::

    C4 = qsymm.rotation(1/4, [1, 0, 0])
    C3 = qsymm.rotation(1/3, [1, 1, 1])
    TR = qsymm.time_reversal(3)

    set(sg) == qsymm.groups.generate_group({C3, C4, TR})


Adding more elements
--------------------
Now we're going to add a 2-fold degeneracy to the above hamiltonian by taking the
kroneker product with the 2x2 identity:

.. jupyter-execute::

    ham_degenerate = "kron(eye(2), " + ham + ")"
    H_degenerate = qsymm.Model(ham_degenerate)

If we look for the symmetries in the same way as previously we will see that the above
modification adds a continuous ``SU(2)`` symmetry:

.. jupyter-execute::

    sg, cg = qsymm.symmetries(H_degenerate, cubic_group)

.. jupyter-execute::

    cg

If we instead add a particle-hole degree of freedom by using :math:`σ_z` instead of the
2x2 identity then we instead get a continuous ``U(1)`` (charge conservation) symmetry:

.. jupyter-execute::

    ham_ph = "kron(sigma_z, " + ham + ")"
    H_ph = qsymm.Model(ham_ph)
    sg, cg = qsymm.symmetries(H_ph, cubic_group)
    print(len(cg))

.. jupyter-execute::

    cg


Rashba Hamiltonian with :math:`J = 3/2`
---------------------------------------

.. jupyter-execute::

    J_x, J_y, J_z = qsymm.groups.spin_matrices(3/2)
    ham_rashba = (
        "hbar^2 / (2 * m) * (k_x**2 + k_y**2 + k_z**2) * eye(4) +" +
        "alpha * J_x * k_x + alpha * J_y * k_y + alpha * J_z * k_z"
    )
    H_rashba = qsymm.Model(ham_rashba, locals=dict(J_x=J_x, J_y=J_y, J_z=J_z))
    H_rashba.tosympy(nsimplify=True)

Note in the above that we had to tell the `~qsymm.model.Model` about the :math:`J_x`,
:math:`J_y` and :math:`J_z` matrices by passing a dictionary ``locals``,
which maps symbols in ``ham_rashba`` to the corresponding matrices.

We again find the symmetries of this model, starting the search from the full cubic group:

.. jupyter-execute::

    sg, cg = qsymm.symmetries(H_rashba, cubic_group)

The symmetry group should be the same as the full cubic group without inversion, plus time-reversal:

.. jupyter-execute::

    set(sg) == qsymm.groups.generate_group({C4, C3, TR})


Bloch Hamiltonian
-----------------
Let's start by defining the hexagonal point group in 2D:

.. jupyter-execute::

    hex_group_2D = qsymm.groups.hexagonal()

Next we define a model for a single-band tight-binding model with nearest-neighbor hopping
on a triangular lattice:

.. jupyter-execute::

    ham_tri = 'm * (cos(k_x) + cos(1/2*k_x + sqrt(3)/2*k_y) + cos(-1/2*k_x + sqrt(3)/2*k_y))'
    display(qsymm.sympify(ham_tri))

.. jupyter-execute::

    H_tri = qsymm.Model(ham_tri, momenta=['k_x', 'k_y'])

We find the symmetries, starting from the full hexagonal point group:

.. jupyter-execute::

    sg, cg = qsymm.symmetries(H_tri, hex_group_2D)

and verify that the point group of the model is the hexagonal group, without particle-hole
symmetry:

.. jupyter-execute::

    set(sg) == qsymm.groups.hexagonal(ph=False)


Continuous rotation symmetry
----------------------------
The Rashba Hamiltonian we defined at the start of this tutorial actually has full rotation invariance:

.. jupyter-execute::

    display(qsymm.sympify(ham))
    pg, cg = qsymm.symmetries(H, continuous_rotations=True, prettify=True)

.. jupyter-execute::

    cg


:math:`k \cdot p` Hamiltonian
-----------------------------
Here we start from an 8x8 :math:`k \cdot p` Hamiltonian that models a zinc-blende semiconductor:

.. jupyter-execute::

    ham_kp = 'Matrix([[hbar**2*k_x*gamma_0*k_x/(2*m_0)+hbar**2*k_y*gamma_0*k_y/(2*m_0)+hbar**2*k_z*gamma_0*k_z/(2*m_0)+E_0+E_v,0,-sqrt(2)*P*k_x/2-sqrt(2)*I*P*k_y/2,sqrt(6)*P*k_z/3,sqrt(6)*P*k_x/6-sqrt(6)*I*P*k_y/6,0,-sqrt(3)*P*k_z/3,-sqrt(3)*P*k_x/3+sqrt(3)*I*P*k_y/3],[0,hbar**2*k_x*gamma_0*k_x/(2*m_0)+hbar**2*k_y*gamma_0*k_y/(2*m_0)+hbar**2*k_z*gamma_0*k_z/(2*m_0)+E_0+E_v,0,-sqrt(6)*P*k_x/6-sqrt(6)*I*P*k_y/6,sqrt(6)*P*k_z/3,sqrt(2)*P*k_x/2-sqrt(2)*I*P*k_y/2,-sqrt(3)*P*k_x/3-sqrt(3)*I*P*k_y/3,sqrt(3)*P*k_z/3],[-sqrt(2)*k_x*P/2+sqrt(2)*I*k_y*P/2,0,-hbar**2*k_x*gamma_1*k_x/(2*m_0)-hbar**2*k_x*gamma_2*k_x/(2*m_0)-I*hbar**2*k_x*k_y/(2*m_0)-3*I*hbar**2*k_x*kappa*k_y/(2*m_0)-hbar**2*k_y*gamma_1*k_y/(2*m_0)-hbar**2*k_y*gamma_2*k_y/(2*m_0)+I*hbar**2*k_y*k_x/(2*m_0)+3*I*hbar**2*k_y*kappa*k_x/(2*m_0)-hbar**2*k_z*gamma_1*k_z/(2*m_0)+hbar**2*k_z*gamma_2*k_z/m_0+E_v,sqrt(3)*hbar**2*k_x*gamma_3*k_z/(2*m_0)+sqrt(3)*hbar**2*k_x*k_z/(6*m_0)+sqrt(3)*hbar**2*k_x*kappa*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*gamma_3*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*k_z/(6*m_0)-sqrt(3)*I*hbar**2*k_y*kappa*k_z/(2*m_0)+sqrt(3)*hbar**2*k_z*gamma_3*k_x/(2*m_0)-sqrt(3)*I*hbar**2*k_z*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_z*k_x/(6*m_0)+sqrt(3)*I*hbar**2*k_z*k_y/(6*m_0)-sqrt(3)*hbar**2*k_z*kappa*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*kappa*k_y/(2*m_0),sqrt(3)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(3)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_y*gamma_2*k_y/(2*m_0)-sqrt(3)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),0,-sqrt(6)*hbar**2*k_x*gamma_3*k_z/(4*m_0)-sqrt(6)*hbar**2*k_x*k_z/(12*m_0)-sqrt(6)*hbar**2*k_x*kappa*k_z/(4*m_0)+sqrt(6)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)+sqrt(6)*I*hbar**2*k_y*k_z/(12*m_0)+sqrt(6)*I*hbar**2*k_y*kappa*k_z/(4*m_0)-sqrt(6)*hbar**2*k_z*gamma_3*k_x/(4*m_0)+sqrt(6)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)+sqrt(6)*hbar**2*k_z*k_x/(12*m_0)-sqrt(6)*I*hbar**2*k_z*k_y/(12*m_0)+sqrt(6)*hbar**2*k_z*kappa*k_x/(4*m_0)-sqrt(6)*I*hbar**2*k_z*kappa*k_y/(4*m_0),-sqrt(6)*hbar**2*k_x*gamma_2*k_x/(2*m_0)+sqrt(6)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)+sqrt(6)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(6)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0)],[sqrt(6)*k_z*P/3,-sqrt(6)*k_x*P/6+sqrt(6)*I*k_y*P/6,sqrt(3)*hbar**2*k_x*gamma_3*k_z/(2*m_0)-sqrt(3)*hbar**2*k_x*k_z/(6*m_0)-sqrt(3)*hbar**2*k_x*kappa*k_z/(2*m_0)+sqrt(3)*I*hbar**2*k_y*gamma_3*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*k_z/(6*m_0)-sqrt(3)*I*hbar**2*k_y*kappa*k_z/(2*m_0)+sqrt(3)*hbar**2*k_z*gamma_3*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*gamma_3*k_y/(2*m_0)+sqrt(3)*hbar**2*k_z*k_x/(6*m_0)+sqrt(3)*I*hbar**2*k_z*k_y/(6*m_0)+sqrt(3)*hbar**2*k_z*kappa*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*kappa*k_y/(2*m_0),-hbar**2*k_x*gamma_1*k_x/(2*m_0)+hbar**2*k_x*gamma_2*k_x/(2*m_0)-I*hbar**2*k_x*k_y/(6*m_0)-I*hbar**2*k_x*kappa*k_y/(2*m_0)-hbar**2*k_y*gamma_1*k_y/(2*m_0)+hbar**2*k_y*gamma_2*k_y/(2*m_0)+I*hbar**2*k_y*k_x/(6*m_0)+I*hbar**2*k_y*kappa*k_x/(2*m_0)-hbar**2*k_z*gamma_1*k_z/(2*m_0)-hbar**2*k_z*gamma_2*k_z/m_0+E_v,hbar**2*k_x*k_z/(3*m_0)+hbar**2*k_x*kappa*k_z/m_0-I*hbar**2*k_y*k_z/(3*m_0)-I*hbar**2*k_y*kappa*k_z/m_0-hbar**2*k_z*k_x/(3*m_0)+I*hbar**2*k_z*k_y/(3*m_0)-hbar**2*k_z*kappa*k_x/m_0+I*hbar**2*k_z*kappa*k_y/m_0,sqrt(3)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(3)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_y*gamma_2*k_y/(2*m_0)-sqrt(3)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),-sqrt(2)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(2)*I*hbar**2*k_x*k_y/(6*m_0)-sqrt(2)*I*hbar**2*k_x*kappa*k_y/(2*m_0)-sqrt(2)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(2)*I*hbar**2*k_y*k_x/(6*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_x/(2*m_0)+sqrt(2)*hbar**2*k_z*gamma_2*k_z/m_0,3*sqrt(2)*hbar**2*k_x*gamma_3*k_z/(4*m_0)-sqrt(2)*hbar**2*k_x*k_z/(12*m_0)-sqrt(2)*hbar**2*k_x*kappa*k_z/(4*m_0)-3*sqrt(2)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)+sqrt(2)*I*hbar**2*k_y*k_z/(12*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_z/(4*m_0)+3*sqrt(2)*hbar**2*k_z*gamma_3*k_x/(4*m_0)-3*sqrt(2)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)+sqrt(2)*hbar**2*k_z*k_x/(12*m_0)-sqrt(2)*I*hbar**2*k_z*k_y/(12*m_0)+sqrt(2)*hbar**2*k_z*kappa*k_x/(4*m_0)-sqrt(2)*I*hbar**2*k_z*kappa*k_y/(4*m_0)],[sqrt(6)*k_x*P/6+sqrt(6)*I*k_y*P/6,sqrt(6)*k_z*P/3,sqrt(3)*hbar**2*k_x*gamma_2*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(3)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),-hbar**2*k_x*k_z/(3*m_0)-hbar**2*k_x*kappa*k_z/m_0-I*hbar**2*k_y*k_z/(3*m_0)-I*hbar**2*k_y*kappa*k_z/m_0+hbar**2*k_z*k_x/(3*m_0)+I*hbar**2*k_z*k_y/(3*m_0)+hbar**2*k_z*kappa*k_x/m_0+I*hbar**2*k_z*kappa*k_y/m_0,-hbar**2*k_x*gamma_1*k_x/(2*m_0)+hbar**2*k_x*gamma_2*k_x/(2*m_0)+I*hbar**2*k_x*k_y/(6*m_0)+I*hbar**2*k_x*kappa*k_y/(2*m_0)-hbar**2*k_y*gamma_1*k_y/(2*m_0)+hbar**2*k_y*gamma_2*k_y/(2*m_0)-I*hbar**2*k_y*k_x/(6*m_0)-I*hbar**2*k_y*kappa*k_x/(2*m_0)-hbar**2*k_z*gamma_1*k_z/(2*m_0)-hbar**2*k_z*gamma_2*k_z/m_0+E_v,-sqrt(3)*hbar**2*k_x*gamma_3*k_z/(2*m_0)+sqrt(3)*hbar**2*k_x*k_z/(6*m_0)+sqrt(3)*hbar**2*k_x*kappa*k_z/(2*m_0)+sqrt(3)*I*hbar**2*k_y*gamma_3*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*k_z/(6*m_0)-sqrt(3)*I*hbar**2*k_y*kappa*k_z/(2*m_0)-sqrt(3)*hbar**2*k_z*gamma_3*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_z*k_x/(6*m_0)+sqrt(3)*I*hbar**2*k_z*k_y/(6*m_0)-sqrt(3)*hbar**2*k_z*kappa*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*kappa*k_y/(2*m_0),3*sqrt(2)*hbar**2*k_x*gamma_3*k_z/(4*m_0)-sqrt(2)*hbar**2*k_x*k_z/(12*m_0)-sqrt(2)*hbar**2*k_x*kappa*k_z/(4*m_0)+3*sqrt(2)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)-sqrt(2)*I*hbar**2*k_y*k_z/(12*m_0)-sqrt(2)*I*hbar**2*k_y*kappa*k_z/(4*m_0)+3*sqrt(2)*hbar**2*k_z*gamma_3*k_x/(4*m_0)+3*sqrt(2)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)+sqrt(2)*hbar**2*k_z*k_x/(12*m_0)+sqrt(2)*I*hbar**2*k_z*k_y/(12*m_0)+sqrt(2)*hbar**2*k_z*kappa*k_x/(4*m_0)+sqrt(2)*I*hbar**2*k_z*kappa*k_y/(4*m_0),sqrt(2)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(2)*I*hbar**2*k_x*k_y/(6*m_0)-sqrt(2)*I*hbar**2*k_x*kappa*k_y/(2*m_0)+sqrt(2)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(2)*I*hbar**2*k_y*k_x/(6*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_x/(2*m_0)-sqrt(2)*hbar**2*k_z*gamma_2*k_z/m_0],[0,sqrt(2)*k_x*P/2+sqrt(2)*I*k_y*P/2,0,sqrt(3)*hbar**2*k_x*gamma_2*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(3)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(3)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),-sqrt(3)*hbar**2*k_x*gamma_3*k_z/(2*m_0)-sqrt(3)*hbar**2*k_x*k_z/(6*m_0)-sqrt(3)*hbar**2*k_x*kappa*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*gamma_3*k_z/(2*m_0)-sqrt(3)*I*hbar**2*k_y*k_z/(6*m_0)-sqrt(3)*I*hbar**2*k_y*kappa*k_z/(2*m_0)-sqrt(3)*hbar**2*k_z*gamma_3*k_x/(2*m_0)-sqrt(3)*I*hbar**2*k_z*gamma_3*k_y/(2*m_0)+sqrt(3)*hbar**2*k_z*k_x/(6*m_0)+sqrt(3)*I*hbar**2*k_z*k_y/(6*m_0)+sqrt(3)*hbar**2*k_z*kappa*k_x/(2*m_0)+sqrt(3)*I*hbar**2*k_z*kappa*k_y/(2*m_0),-hbar**2*k_x*gamma_1*k_x/(2*m_0)-hbar**2*k_x*gamma_2*k_x/(2*m_0)+I*hbar**2*k_x*k_y/(2*m_0)+3*I*hbar**2*k_x*kappa*k_y/(2*m_0)-hbar**2*k_y*gamma_1*k_y/(2*m_0)-hbar**2*k_y*gamma_2*k_y/(2*m_0)-I*hbar**2*k_y*k_x/(2*m_0)-3*I*hbar**2*k_y*kappa*k_x/(2*m_0)-hbar**2*k_z*gamma_1*k_z/(2*m_0)+hbar**2*k_z*gamma_2*k_z/m_0+E_v,sqrt(6)*hbar**2*k_x*gamma_2*k_x/(2*m_0)+sqrt(6)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(6)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(6)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),-sqrt(6)*hbar**2*k_x*gamma_3*k_z/(4*m_0)-sqrt(6)*hbar**2*k_x*k_z/(12*m_0)-sqrt(6)*hbar**2*k_x*kappa*k_z/(4*m_0)-sqrt(6)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)-sqrt(6)*I*hbar**2*k_y*k_z/(12*m_0)-sqrt(6)*I*hbar**2*k_y*kappa*k_z/(4*m_0)-sqrt(6)*hbar**2*k_z*gamma_3*k_x/(4*m_0)-sqrt(6)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)+sqrt(6)*hbar**2*k_z*k_x/(12*m_0)+sqrt(6)*I*hbar**2*k_z*k_y/(12*m_0)+sqrt(6)*hbar**2*k_z*kappa*k_x/(4*m_0)+sqrt(6)*I*hbar**2*k_z*kappa*k_y/(4*m_0)],[-sqrt(3)*k_z*P/3,-sqrt(3)*k_x*P/3+sqrt(3)*I*k_y*P/3,-sqrt(6)*hbar**2*k_x*gamma_3*k_z/(4*m_0)+sqrt(6)*hbar**2*k_x*k_z/(12*m_0)+sqrt(6)*hbar**2*k_x*kappa*k_z/(4*m_0)-sqrt(6)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)+sqrt(6)*I*hbar**2*k_y*k_z/(12*m_0)+sqrt(6)*I*hbar**2*k_y*kappa*k_z/(4*m_0)-sqrt(6)*hbar**2*k_z*gamma_3*k_x/(4*m_0)-sqrt(6)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)-sqrt(6)*hbar**2*k_z*k_x/(12*m_0)-sqrt(6)*I*hbar**2*k_z*k_y/(12*m_0)-sqrt(6)*hbar**2*k_z*kappa*k_x/(4*m_0)-sqrt(6)*I*hbar**2*k_z*kappa*k_y/(4*m_0),-sqrt(2)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(2)*I*hbar**2*k_x*k_y/(6*m_0)-sqrt(2)*I*hbar**2*k_x*kappa*k_y/(2*m_0)-sqrt(2)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(2)*I*hbar**2*k_y*k_x/(6*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_x/(2*m_0)+sqrt(2)*hbar**2*k_z*gamma_2*k_z/m_0,3*sqrt(2)*hbar**2*k_x*gamma_3*k_z/(4*m_0)+sqrt(2)*hbar**2*k_x*k_z/(12*m_0)+sqrt(2)*hbar**2*k_x*kappa*k_z/(4*m_0)-3*sqrt(2)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)-sqrt(2)*I*hbar**2*k_y*k_z/(12*m_0)-sqrt(2)*I*hbar**2*k_y*kappa*k_z/(4*m_0)+3*sqrt(2)*hbar**2*k_z*gamma_3*k_x/(4*m_0)-3*sqrt(2)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)-sqrt(2)*hbar**2*k_z*k_x/(12*m_0)+sqrt(2)*I*hbar**2*k_z*k_y/(12*m_0)-sqrt(2)*hbar**2*k_z*kappa*k_x/(4*m_0)+sqrt(2)*I*hbar**2*k_z*kappa*k_y/(4*m_0),sqrt(6)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(6)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)-sqrt(6)*hbar**2*k_y*gamma_2*k_y/(2*m_0)-sqrt(6)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),-hbar**2*k_x*gamma_1*k_x/(2*m_0)-I*hbar**2*k_x*k_y/(3*m_0)-I*hbar**2*k_x*kappa*k_y/m_0-hbar**2*k_y*gamma_1*k_y/(2*m_0)+I*hbar**2*k_y*k_x/(3*m_0)+I*hbar**2*k_y*kappa*k_x/m_0-hbar**2*k_z*gamma_1*k_z/(2*m_0)-Delta+E_v,hbar**2*k_x*k_z/(3*m_0)+hbar**2*k_x*kappa*k_z/m_0-I*hbar**2*k_y*k_z/(3*m_0)-I*hbar**2*k_y*kappa*k_z/m_0-hbar**2*k_z*k_x/(3*m_0)+I*hbar**2*k_z*k_y/(3*m_0)-hbar**2*k_z*kappa*k_x/m_0+I*hbar**2*k_z*kappa*k_y/m_0],[-sqrt(3)*k_x*P/3-sqrt(3)*I*k_y*P/3,sqrt(3)*k_z*P/3,-sqrt(6)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(6)*I*hbar**2*k_x*gamma_3*k_y/(2*m_0)+sqrt(6)*hbar**2*k_y*gamma_2*k_y/(2*m_0)-sqrt(6)*I*hbar**2*k_y*gamma_3*k_x/(2*m_0),3*sqrt(2)*hbar**2*k_x*gamma_3*k_z/(4*m_0)+sqrt(2)*hbar**2*k_x*k_z/(12*m_0)+sqrt(2)*hbar**2*k_x*kappa*k_z/(4*m_0)+3*sqrt(2)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)+sqrt(2)*I*hbar**2*k_y*k_z/(12*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_z/(4*m_0)+3*sqrt(2)*hbar**2*k_z*gamma_3*k_x/(4*m_0)+3*sqrt(2)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)-sqrt(2)*hbar**2*k_z*k_x/(12*m_0)-sqrt(2)*I*hbar**2*k_z*k_y/(12*m_0)-sqrt(2)*hbar**2*k_z*kappa*k_x/(4*m_0)-sqrt(2)*I*hbar**2*k_z*kappa*k_y/(4*m_0),sqrt(2)*hbar**2*k_x*gamma_2*k_x/(2*m_0)-sqrt(2)*I*hbar**2*k_x*k_y/(6*m_0)-sqrt(2)*I*hbar**2*k_x*kappa*k_y/(2*m_0)+sqrt(2)*hbar**2*k_y*gamma_2*k_y/(2*m_0)+sqrt(2)*I*hbar**2*k_y*k_x/(6*m_0)+sqrt(2)*I*hbar**2*k_y*kappa*k_x/(2*m_0)-sqrt(2)*hbar**2*k_z*gamma_2*k_z/m_0,-sqrt(6)*hbar**2*k_x*gamma_3*k_z/(4*m_0)+sqrt(6)*hbar**2*k_x*k_z/(12*m_0)+sqrt(6)*hbar**2*k_x*kappa*k_z/(4*m_0)+sqrt(6)*I*hbar**2*k_y*gamma_3*k_z/(4*m_0)-sqrt(6)*I*hbar**2*k_y*k_z/(12*m_0)-sqrt(6)*I*hbar**2*k_y*kappa*k_z/(4*m_0)-sqrt(6)*hbar**2*k_z*gamma_3*k_x/(4*m_0)+sqrt(6)*I*hbar**2*k_z*gamma_3*k_y/(4*m_0)-sqrt(6)*hbar**2*k_z*k_x/(12*m_0)+sqrt(6)*I*hbar**2*k_z*k_y/(12*m_0)-sqrt(6)*hbar**2*k_z*kappa*k_x/(4*m_0)+sqrt(6)*I*hbar**2*k_z*kappa*k_y/(4*m_0),-hbar**2*k_x*k_z/(3*m_0)-hbar**2*k_x*kappa*k_z/m_0-I*hbar**2*k_y*k_z/(3*m_0)-I*hbar**2*k_y*kappa*k_z/m_0+hbar**2*k_z*k_x/(3*m_0)+I*hbar**2*k_z*k_y/(3*m_0)+hbar**2*k_z*kappa*k_x/m_0+I*hbar**2*k_z*kappa*k_y/m_0,-hbar**2*k_x*gamma_1*k_x/(2*m_0)+I*hbar**2*k_x*k_y/(3*m_0)+I*hbar**2*k_x*kappa*k_y/m_0-hbar**2*k_y*gamma_1*k_y/(2*m_0)-I*hbar**2*k_y*k_x/(3*m_0)-I*hbar**2*k_y*kappa*k_x/m_0-hbar**2*k_z*gamma_1*k_z/(2*m_0)-Delta+E_v]])'
    Hkp = qsymm.Model(ham_kp)

The symmetry is the full cubic group with time reversal; 96 elements:

.. jupyter-execute::

    sg, cg = qsymm.symmetries(Hkp, cubic_group)
    print(len(sg))

If we restrict the parameters we can change the symmetry group:

.. jupyter-execute::

    kp_examples = [
        {'hbar': 1},
        {'hbar': 1, 'P': 0, 'Delta': 0},
        {'hbar': 1, 'P': 0, 'Delta': 0, 'gamma_3': 0},
        {'hbar': 1, 'gamma_2': 'gamma_1', 'gamma_3': 'gamma_1', 'k_z': 0},
        {'hbar': 1, 'P': 0, 'Delta': 0, 'gamma_3': 0},
        {'hbar': 1, 'Delta': 0, 'k_z': 0},
    ]

    for subs in kp_examples:
        Hkp = qsymm.Model(ham_kp, locals=subs)
        sg, cg = qsymm.symmetries(Hkp, cubic_group)
        print(subs, len(sg), len(cg))
