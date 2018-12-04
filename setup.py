#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys


if sys.version_info < (3, 5):
    print('Qsymm requires Python 3.5 or above.')
    sys.exit(1)


install_requires = [
    'numpy>=1.13',  # because we use __array_ufunc__
    'scipy>=1.1.0', # because we use reshape for sparse matrices
    'sympy>=1.1',
    'tinyarray',
    'notebook',
]

extras_require = {
    'kwant': ['kwant'],
}


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


setup(
    name='qsymm',
    description='Symmetry finder and symmetric Hamiltonian generator',
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
    version=version,
    cmdclass=cmdclass,
)
