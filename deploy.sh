python setup.py sdist;
python setup.py test;
git add .;
git commit -m "Deployment.";
git push;
git pull;
twine upload dist/*;
pip uninstall flyqma;
pip install flyqma;

echo 'Finished deploying new version to PyPI, pushed update to github.'
