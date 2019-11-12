#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys


if sys.version_info < (3, 5):
    print('Qsymm requires Python 3.5 or above.')
    sys.exit(1)


# These requirements should not be updated to versions more
# recent than required by Kwant:
# https://gitlab.kwant-project.org/kwant/kwant/blob/master/setup.py#L584
# because we want to remain interoperable
install_requires = [
    'numpy>=1.13',  # because we use __array_ufunc__
    'scipy>=0.19',
    'sympy>=1.1',
    'tinyarray',
]

extras_require = {
    'kwant': ['kwant'],
}

tests_require = [
    'pytest',
]

setup_requires = [
    'pytest-runner',
]


# Loads version.py module without importing the whole package.
def get_version_and_cmdclass(package_path):
    import os
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('version',
                                   os.path.join(package_path, '_version.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass('qsymm')


with open("README.md") as f:
    long_description = f.read()


setup(
    name='qsymm',
    description='Symmetry finder and symmetric Hamiltonian generator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab.kwant-project.org/qt/qsymm',
    author='Qsymm authors',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages('.'),
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    setup_requires=setup_requires,
    version=version,
    cmdclass=cmdclass,
)
