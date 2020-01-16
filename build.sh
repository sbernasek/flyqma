python setup.py sdist;
twine upload dist/*;
pip uninstall flyqma;
pip install flyqma;
