#!/bin/bash

for file in $*
do
  base=`basename $file .bag`
  echo 'Reannotating' $file
  python annotate.py $file $base-center.npz --reannotate $base-center.npz
done

