#! /bin/bash

ROOT='data'
mkdir -p $ROOT
cd $ROOT
wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xzf ShapeNetVox32.tgz
