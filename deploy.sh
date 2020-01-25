python setup.py sdist;
git add .;
git commit -m "Deployment.";
git push;
git pull;
twine upload dist/*;
pip uninstall flyqma;
pip install flyqma;
