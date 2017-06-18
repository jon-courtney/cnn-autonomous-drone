#!/bin/bash

for file in $*
do
  python play.py $file
done

