# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2019-10-17
### Fixes
- Adding zero to a model is now idempotent

## [1.2.1] - 2019-09-02
### Added
- Add CHANGELOG.md

### Fixed
+ Restore compatibility with Python 3.5

## [1.2.0] - 2019-08-30
### Added
- Factory functions for discrete symmetries: `time_reversal`, `particle_hole`, `chiral`, `inversion`, `rotation`, `mirror`.
- Better representation of `PointGroupElement`s and `ContinuousGroupGenerator`, using `_repr_pretty_` and `_repr_latex_`. Remove `print_PG_elements` and update example notebooks.
- Implemented new functionality in `Model`:
 + Implement `__matmul__` (`@`).
 + Support of sparse matrices and `LinearOperator` as values.
 + Consistent support of scalar valued `Model`s.
 + Add `keep` to only keep track of certain symbolic coefficients.
 + More options and more transparent initialization, allow string keys which are automatically symmpified by default.
 + Several new utility functions, such as `trace`, `reshape`, `allclose` etc.

### Changed
- Slight change to the internal representation of `PointGroupElements`, allow mixing integer tinyarray with either sympy matrix or floating point tinyarray in the rotation part, but do not allow mixing the latter two representations. This removes the need to have two different representations for simple operators. Optimize multiplication by caching.
- Changes in the API of `Model`:
 - Add `format` attribute to keep track of the type of data entries (i.e scalar, dense or sparse array).
 - Change the behaviour of `*` for `Model` objects, matrix multiplication only works with `@` from now on. This breaks backward compatibility, a few uses in the qsymm code base were fixed.
 - Stop supporting a sequence of ints as `momenta`, instead string keys (e.g. `'k_x'`) or `sympy` expressions can be used. Certain other, rarely used ways to initialize `Model` don't work anymore, some tests were changed accordingly.
 - Stop rounding and removing small entries from `Model` objects. This means that testing that the `Model` is empty is not a good test anymore to see if it is approximately zero. Use `Model.allclose` instead.
- Optimizations in `Model`:
 + Remove unnecessary deep copying, which was slow.
 + Optimize the implementation of arithmetic operations to minimize number of loops and function calls.
 + Fast initialization by making restructuring optional when initialized with a dict.
 + Clean up the code of `BlochModel` to utilize improvements in `Model`.
 + Update symmetry finder to work with sparse models.

### Deprecated
- Deprecate initializing empty `Model` without providing `shape` and `format`.

## [1.1.0] - 2018-12-05
Many of these changes were made in anticipation of integrating `qsymm` with `kwant`,
allowing conversion between the Hamiltonian formats used.

### Added
+ `BlochModel` as a subclass of `Model` in order to store Bloch Hamiltonians
  with floating point hopping vectors.
+ `bravais_point_group` to find the point group of a Bravais lattice using the
  translation vectors only. This is intended for generating candidades for `symmetries`.

### Changed
+ Change the way equality of `Model`s is tested, the current implementation treats
  tolerances more consistently, this fixes some bugs in symmetry finding.
+ Allow using floating point rotation matrices in `PointGroupElement`.

### Fixed
+ Several bugs and internal code restructuring.
