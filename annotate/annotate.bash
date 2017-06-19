#!/bin/bash

for file in $*
do
  base=`basename $file -jpg.npz`
  echo $base-hsv.npz
  python annotate.py $file $base-hsv.npz
done

