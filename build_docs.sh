#!/bin/bash

build_docs ()
{
  mv ./docs ./html
  cd ./html
  cmd /c make.bat html
  cd -
  mv ./html ./docs
}

build_docs
