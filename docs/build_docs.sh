#!/bin/sh

git add . ;
git commit -m "Updated documentation." ;
git push; git pull;
sphinx-build -a source ./ ;
git add . ;
git commit -m "Updated documentation." ;
git push;
git pull;

echo 'Finished building documentation, pushed updated to github.'
