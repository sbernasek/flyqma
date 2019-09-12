#!/bin/sh

workon clones;
sphinx-build -a source ./ ;
git add . ;
git commit -m "updated docs" ;
git push;
git pull;

echo 'COMPLETE.'
