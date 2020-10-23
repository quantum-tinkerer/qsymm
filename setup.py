#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# These requirements should not be updated to versions more
# recent than required by Kwant:
# https://gitlab.kwant-project.org/kwant/kwant/blob/master/setup.py#L584
# because we want to remain interoperable
install_requires = [
    'numpy>=1.16.2',
    'scipy>=1.1',
    'sympy>=1.3.0',
    'tinyarray>=1.2.2',
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
    python_requires='>=3.7',
    version=version,
    cmdclass=cmdclass,
)
