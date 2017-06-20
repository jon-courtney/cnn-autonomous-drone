#!/bin/bash

for file in $*
do
  base=`basename $file .bag`
  python extract.py $file $base-jpg.npz
done

