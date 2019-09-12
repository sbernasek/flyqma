import sys, os
from subprocess import call
from distutils.core import setup
from distutils.command.install import install as _install
from setuptools import find_packages


def _post_install(dir):
    """ Install pomegranate without dependencies (due to networkx conflict) """
    prefix = [sys.executable, '-m']
    cmd = "pip install --user pomegranate==0.10.0 --no-deps".split()
    call(prefix+cmd, cwd=os.path.join(dir, 'flyqma'))


class install(_install):
    def run(self):
        _install.run(self)
        self.execute(_post_install, (self.install_lib,),
                     msg="Running post install task")

setup(
    name='flyqma',
    version='0.1.0',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://github.com/sebastianbernasek/flyqma',
    license='MIT',
    description='Quantitative mosaic analysis of clonal subpopulations in the Drosophila eye.',
    long_description=open('README.md').read(),
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
    ],
    cmdclass={'install': install},
)
