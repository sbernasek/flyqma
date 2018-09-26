from distutils.core import setup
from setuptools import find_packages

setup(
    name='clones',
    version='0.1.0',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://github.com/sebastianbernasek/clones',
    license='MIT',
    description='Quantitative analysis of clonal subpopulations in the Drosophila eye.',
    long_description=open('README.md').read(),
    install_requires=[
        "tifffile >= 0.15.0",
        "scipy >= 1.1.0",
        "scikit-image >= 0.14.0",
        "scikit-learn >= 0.19.2",
        "statsmodels == 0.9.0",
        "pandas >= 0.23.3",
        #"pillow >= 5.2.0",
        #"tables >= 3.4.4",
        "seaborn >= 0.9.0",
        "networkx >= 2.1",
        "infomap >= 1.0.0b8"
    ],
)

