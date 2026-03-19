# Making a qsymm release

## Ensure that all tests pass

At minimum, run:

```bash
pixi run -e precommit pre-commit run --all-files
pixi run -e minimal tests
pixi run -e latest tests
pixi run docs-build
```

## Update the changelog

Change the top `unreleased` line to the new version number and add the date, e.g.

```
## [1.2.5] - 2019-11-11
```

Add a new `unreleased` line to the top of the changelog, e.g.

```
## [unreleased]

## [1.2.5] - 2019-11-11
```

## Tag the release

Make an **annotated** tag for the release. The tag must be the version number prefixed by the letter 'v':
```
git tag v<version> -m "version <version>"
```

## Build a source tarball and wheels and test it

Build with the dedicated publish environment:

```bash
rm -fr build dist
pixi run -e publish build
```

This creates files in `dist/`. It is a good idea to check that the metadata is
plausible and that the expected artifacts were created.

It is also a good idea to unpack the source distribution and check that the
tests run:

```bash
tar xzf dist/qsymm*.tar.gz
cd qsymm-*
pixi run -e minimal test
```

## Publish the release

Publishing is performed by GitLab CI from release tags.

### Push the release tag

```bash
git push origin v<version>
```

Pushing `v<version>` triggers the `publish to pypi` job in CI.

### Optional: publish a test release

To exercise the release pipeline against TestPyPI, push a tag that matches the
test-release rule, for example:

```bash
git tag v<version>.post1+test -m "test release <version>"
git push origin v<version>.post1+test
```

This triggers the `publish to test pypi` job in CI.

### Create conda-forge package

Some time (typically minutes/hours) after making the PyPI release a pull
request will automatically be opened on the
[Qsymm feedstock](https://github.com/conda-forge/qsymm-feedstock/) repository.

This pull request should be checked to make sure the tests pass and can then
be merged. This will make a new release of the Qsymm conda package on conda-forge.
