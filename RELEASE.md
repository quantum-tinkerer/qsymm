# Making a qsymm release

## Validate the branch in CI

Before tagging a release, require a green GitLab pipeline for the release
branch or merge request.

The release is blocked until the following CI jobs succeed:

- `run tests`
- `run coverage`
- `run pre-commit`
- `build docs`

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

## Optional: reproduce the release build locally

The release build is performed in GitLab CI. A local build is only needed for
debugging the packaging setup or investigating a failing release job.

To reproduce the CI build locally:

```bash
rm -fr build dist
pixi run -e publish build
```

## Publish the release

Publishing is performed by GitLab CI from release tags.

### Push the release tag

```bash
git push origin v<version>
```

Pushing `v<version>` triggers the `publish to pypi` job in CI.

The release is only complete once that CI job succeeds.

### Optional: publish a test release

To exercise the release pipeline against TestPyPI, push a tag that matches the
test-release rule, for example:

```bash
git tag v<version>.post1+test -m "test release <version>"
git push origin v<version>.post1+test
```

This triggers the `publish to test pypi` job in CI.

Use this when you want to validate the release pipeline itself before pushing
the final release tag.

### Create conda-forge package

Some time (typically minutes/hours) after making the PyPI release a pull
request will automatically be opened on the
[Qsymm feedstock](https://github.com/conda-forge/qsymm-feedstock/) repository.

This pull request should be checked to make sure the tests pass and can then
be merged. This will make a new release of the Qsymm conda package on conda-forge.
