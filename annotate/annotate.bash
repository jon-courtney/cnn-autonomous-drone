#!/bin/bash

for file in $*
do
  base=`basename $file .bag`
  python annotate.py $file $base-hsv.npz
done

