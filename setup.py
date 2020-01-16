import sys, os
from subprocess import call
from distutils.core import setup
from distutils.command.install import install as _install
from setuptools import find_packages


setup(
    name='flyqma',
    version='0.1',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.edu',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://sebastianbernasek.github.io/flyqma/',
    license='MIT',
    description='Quantitative mosaic analysis of Drosophila imaginal discs.',
    long_description=open('README.md').read(),
    python_requires='>=3',
    install_requires=[
        "tifffile >= 0.15.0",
        "scipy >= 1.1.0",
        "scikit-image >= 0.14.0",
        "scikit-learn >= 0.19.2",
        "statsmodels == 0.9.0",
        "pandas >= 0.23.3",
        #"pillow >= 5.2.0",
        "tables >= 3.4.4",
        "seaborn >= 0.9.0",
        "networkx >= 2.1",
        "infomap >= 1.0.0b8",
        "dill >= 0.2.8.2",
        "cython >= 0.22.1",
        "joblib >= 0.9.0b4",
        "pyyaml"
    ]
)
