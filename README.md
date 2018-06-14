# Qsymm

**Symmetry finder and symmetric Hamiltonian generator**

`qsymm` is an [open-source](LICENSE) Python library that makes symmetry analysis simple.
You can generate model Hamiltonians from symmetry constraints or find out the full symmetry group of your Hamiltonian with just a few lines of code.

Check out the introductory [example notebook `basics.ipynb`](basics.ipynb) to see examples of how to use `qsymm`.


## Implemented algorithms
The two core concepts in `qsymm` are _Hamiltonian families_ (Hamiltonians that may depend on free parameters) and _symmetries_. We provide powerful classes to handle these:

+ `Model` is used to store symbolic Hamiltonians that may depend on momenta and other free parameters. We use `sympy` for symbolic manipulation, but our implementation utilizes `numpy` arrays for efficient calculations with matrix valued functions.

+ `PointGroupElement` and `ContinuousGroupGenerator` are used to store symmetry operators. Besides the ability to combine symmetries, they can also act on `Model`s.

We implement algorithms that form a two-way connection between Hamiltonian families and symmetries.

+ Symmetry finding is handled by `symmetries`, it takes a `Model` as input and finds all of its symmetries, including conserved quantities, time reversal, particle-hole, and spatial rotation symmetries. See [`symmetry_finder.ipynb`](symmetry_finder.ipynb) and [`kekule.ipynb`](kekule.ipynb) for detailed examples.

+ `continuum_hamiltonian` and `bloch_family` are used to generate __k.p__ or lattice Hamiltonians from symmetry constraints. See [`kdotp_generator.ipynb`](kdotp_generator.ipynb), [`bloch_generator.ipynb`](bloch_generator.ipynb) and [`kekule.ipynb`](kekule.ipynb) for detailed examples.

## Installation
`qsymm` works with Python 3.5 and is available on PyPI:
```bash
pip install qsymm[kwant]
```

The `[kwant]` above will also install [Kwant](https://kwant-project.org/) that we use in some examples.
