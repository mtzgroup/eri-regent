#!/bin/bash

set -eu

LEGION_OPTIONS="\
-ll:cpu 1 -ll:gpu 0 \
-ll:csize 1024 -ll:fsize 1024"

DATA_DIRECTORY="data"

INPUT_FILES="
h2o_6-311g.dat
small-water/h2o_2_6-311g.dat
small-water/h2o_3_6-311g.dat
small-water/h2o_4_6-311g.dat
water-boxes/h2o_5_6-311g.dat
water-boxes/h2o_10_6-311g.dat
"
# water-boxes/h2o_50_6-311g.dat
# water-boxes/h2o_100_6-311g.dat
# water-boxes/h2o_250_6-311g.dat
# water-boxes/h2o_500_6-311g.dat
# water-boxes/h2o_750_6-311g.dat
# water-boxes/h2o_1000_6-311g.dat
# "

for FILE in $INPUT_FILES; do
  echo $FILE
  regent top.rg -i "$DATA_DIRECTORY/$FILE" $LEGION_OPTIONS 2> /dev/null | sed -n "s/Coulomb operator: \(.*\) sec/\1/p"
done
