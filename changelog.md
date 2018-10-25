### Changelog

## Qsymm 1.1

+ Introduce `BlochModel` as a subclass of `Model` in order to store Bloch Hamiltonians with floating point hopping vectors.

+ Change the way equality of `Model`s is tested, the current implementation treats tolerances more consistently, this fixes some bugs in symmetry finding.

+ Allow using floating point rotation matrices in `PointGroupElement`.

+ Add `bravais_point_group` to find the point group of a Bravais lattice using the translation vectors only. This is intended for generating candidades for `symmetries`.

+ Several bugfixes and internal code restructuring.

Many of these changes were made in anticipation of integrating `qsymm` with `kwant`, allowing conversion between the Hamiltonian formats used.
