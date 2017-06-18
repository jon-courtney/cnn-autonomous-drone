#!/bin/bash

for file in $*
do
  base=`basename $file .npz`
  python synthesize.py $file $base-synth.npz
done

