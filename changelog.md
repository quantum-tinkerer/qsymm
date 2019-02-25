# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] -- 2018-12-05
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

