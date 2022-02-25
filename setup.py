
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='friqml',
    version='0.0.1',
    description='Auxiliary routines for the course Quantum machine learning',
    long_description=readme,
    author='Bojan Žunkovič',
    author_email='bojan.zunkovic@fri.uni-lj.si',
    url='https://github.com/znajob/qml-tn.git',
    license=license,
    install_requires=[
        'pennylane',
        'qutip'
    ],
    packages=["friqml"],
)
