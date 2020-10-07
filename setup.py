import sys, os
from subprocess import call
from distutils.core import setup
from distutils.command.install import install as _install
from setuptools import find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='flyqma',
    version='0.4',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.edu',
    packages=find_packages(exclude=('tests', 'scripts', 'validation')),
    scripts=[],
    url='https://sbernasek.github.io/flyqma/',
    license='MIT',
    description='Quantitative mosaic analysis of Drosophila imaginal discs.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3',
    install_requires=[
        "tifffile >= 0.15.0",
        "scipy >= 1.4.0",
        "scikit-image >= 0.14.0",
        "scikit-learn >= 0.19.2",
        "statsmodels >= 0.11.0",
        "pandas >= 0.23.3",
        "tables >= 3.4.4",
        "seaborn >= 0.9.0",
        "networkx >= 2.1",
        "dill >= 0.2.8.2",
        "joblib >= 0.9.0b4",
        "pyyaml",
        #"cython >= 0.22.1",
        #"pillow >= 5.2.0",
        #"infomap >= 1.0.0b8"
    ],
    tests_require=['nose'],
    test_suite='nose.collector'
)
