# Making a qsymm release

## Ensure that all tests pass

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

```
rm -fr build dist
python setup.py sdist bdist_wheel
```

This creates the file `dist/qsymm-<version>.tar.gz`.  It is a good idea to unpack it
and check that the tests run:
```
tar xzf dist/qsymm*.tar.gz
cd qsymm-*
py.test .
```

## Create an empty commit for new development and tag it
```
git commit --allow-empty -m 'start development towards v<version+1>'
git tag -am 'Start development towards v<version+1>' v<version+1>-dev
```

Where `<version+1>` is `<version>` with the minor version incremented
(or major version incremented and minor and patch versions then reset to 0).
This is necessary so that the reported version for any further commits is
`<version+1>-devX` and not `<version>-devX`.


## Publish the release

### Push the tags
```
git push origin v<version> v<version+1>-dev
```

### Upload to PyPI
```
twine upload dist/*
```

### Create conda-forge package

Some time (typically minutes/hours) after making the PyPI release a pull
request will automatically be opened on the
[Qsymm feedstock](https://github.com/conda-forge/qsymm-feedstock/) repository.

This pull request should be checked to make sure the tests pass and can then
be merged. This will make a new release of the Qsymm conda package on conda-forge.
